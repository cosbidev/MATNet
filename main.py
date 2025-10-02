# %%
import json
import os
from time import time

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src import miner, model, utils

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_model(train_loader, val_loader, test_loader, CHECKPOINT_PATH, exp_filename, model_name, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, exp_filename),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=200,
        log_every_n_steps=36,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True),
            LearningRateMonitor("epoch"),
        ],
    )
    os.makedirs(trainer.default_root_dir, exist_ok=True)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"{exp_filename}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyper-parameters
        net = model.MPVForecaster.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducible
        net = model.MPVForecaster(model_name, train_loader, **kwargs)
        trainer.fit(net, train_loader, val_loader)
        # Load the best checkpoint after training
        net = model.MPVForecaster.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(net, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(net, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_loss"], "val": val_result[0]["test_loss"]}

    return model, result


def lunch_experiment(root, exp_setup):
    start = time()
    # Setting the seed
    pl.seed_everything(42)

    exp_filename = utils.create_filename(exp_setup)
    model_name = exp_setup["model_name"]

    hour_on = exp_setup["temporal_ablation"]["hour_on"]
    day_on = exp_setup["temporal_ablation"]["day_on"]
    month_on = exp_setup["temporal_ablation"]["month_on"]

    win_length = exp_setup["sliding_window"]["win_length"]
    step = exp_setup["sliding_window"]["step"]
    time_horizon = exp_setup["sliding_window"]["time_horizon"]

    pv_on = exp_setup["branch_ablation"]["pv_forecast"]
    swx_on = exp_setup["branch_ablation"]["wx_history"]
    fwx_on = exp_setup["branch_ablation"]["wx_forecast"]

    num_temporal = utils.count_true(exp_setup["temporal_ablation"]) * 2

    model_kwargs = {"pv_features": 1,
                    "hw_features": 34 + num_temporal,
                    "fw_features": 34 + num_temporal,
                    "n_steps_in": win_length,
                    "n_steps_out": time_horizon,
                    }

    if model_name == "MATNet":
        num_layers = exp_setup["MATNet_architecture_setup"]["num_layers"]
        fusion = exp_setup["MATNet_architecture_setup"]["fusion"]
        interpolation = exp_setup["MATNet_architecture_setup"]["interpolation"]
        interpolation_factor = exp_setup["MATNet_architecture_setup"]["interpolation_factor"]

        model_kwargs.update({'d_model': 512,
                             'nhead': 8,
                             'num_layers': num_layers,
                             'dim_feedforward': 1024,
                             # 'fusion': fusion,
                             # 'interpolation': interpolation,
                             # 'interp_factor': interpolation_factor,
                             })

        model_architecture = f"{interpolation}InterpFact" \
                             f"{interpolation_factor if interpolation_factor else 0}-" \
                             f"Fus{fusion}-NumLayers{num_layers}"

    else:
        model_kwargs.update({"bidirectional": "Bi" in model_name,
                             })
        model_architecture = model_name.split('_')[1]
        model_kwargs.update(({"recurrent": "LSTM" if "LSTM" in model_architecture else "GRU"}))

    # Loading the training dataset. We need to split it into a training and validation part
    dataset = miner.MVAusgrid(root=root, train=True, plants=None, max_kwp=True, win_length=win_length, step=step,
                              time_horizon=time_horizon, normalize='min-max', scaler=None, eps=1e-5, pv_on=pv_on,
                              swx_on=swx_on, fwx_on=fwx_on, hour_on=hour_on, day_on=day_on,
                              month_on=month_on, plant=None)

    test_set = miner.MVAusgrid(root=root, train=False, plants=None, max_kwp=True, win_length=win_length, step=step,
                               time_horizon=time_horizon, normalize='min-max', scaler=None, eps=1e-5,
                               pv_on=pv_on, swx_on=swx_on, fwx_on=fwx_on, hour_on=hour_on, day_on=day_on,
                               month_on=month_on, plant=None)

    train_set, val_set = data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True,
                                   pin_memory=True, num_workers=16)
    val_loader = data.DataLoader(val_set, batch_size=16, shuffle=False, drop_last=False, num_workers=16)
    test_loader = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, drop_last=False,
                                  num_workers=0)

    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT",
                                     f"saved_models/{model_name.split('_')[0]}/no-pv/{model_architecture}/")

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    mdl, results = train_model(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                               CHECKPOINT_PATH=CHECKPOINT_PATH, exp_filename=exp_filename, lr=0.001,
                               model_name=model_name.split("_")[0],
                               model_kwargs=model_kwargs,
                               )
    end = time()
    print(f"{exp_filename} Finish with:{(end - start) / 60} minutes, results", results)


def main(exp_idx=None):
    root = "./Data"

    with open('./config/matnet_config.json', 'r') as data_file:
        json_data = data_file.read()

    experiment_list = json.loads(json_data)

    if exp_idx is not None:
        lunch_experiment(root, experiment_list[exp_idx])

    else:
        for exp in experiment_list:
            lunch_experiment(root, exp)

        print("All experiments completed")


if __name__ == "__main__":
    main(exp_idx=None)
