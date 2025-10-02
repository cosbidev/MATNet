import os
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src import utils
import re

results = {key: [] for key in ['model_name',
                               'hour_on',
                               'day_on',
                               'month_on',
                               'wx_history',
                               'wx_forecast',
                               'pv_forecast',
                               'win_length',
                               'step',
                               'time_horizon',
                               # 'training_time',
                               'mean_MAE',
                               'std_MAE',
                               'mean_MAPE',
                               'std_MAPE',
                               'mean_MSE',
                               'std_MSE',
                               'mean_RMSE',
                               'std_RMSE',
                               'mean_SMAPE',
                               'std_SMAPE',
                               'mean_WMAPE',
                               'std_WMAPE',
                               'mean_MAAPE',
                               'std_MAAPE',
                               'mean_MDA',
                               'std_MDA',
                               'mean_MASE',
                               'std_MASE',
                               'best_val_mean_MAE',
                               'best_val_std_MAE',
                               'best_val_mean_MAPE',
                               'best_val_std_MAPE',
                               'best_val_mean_MSE',
                               'best_val_std_MSE',
                               'best_val_mean_RMSE',
                               'best_val_std_RMSE',
                               'best_val_mean_SMAPE',
                               'best_val_std_SMAPE',
                               'best_val_mean_WMAPE',
                               'best_val_std_WMAPE',
                               'best_val_mean_MAAPE',
                               'best_val_std_MAAPE',
                               'best_val_mean_MDA',
                               'best_val_std_MDA',
                               'best_val_mean_MASE',
                               'best_val_std_MASE'
                               ]}


def main():
    weather_data = "wx_data.xlsx"
    i = 1
    for root, dirs, files in os.walk("./saved_models/MATNet/AdaptiveInterpFact0-FusSoftAttention-NumLayers3", topdown=False):
        for name in dirs:
            if "version" in name:
                experiment_path = os.path.join(root, name)
                experiment_name = list(Path(experiment_path).parts)[-3]
                experiment_setup = utils.filename2setup(experiment_name)

                #event_acc = EventAccumulator(experiment_path)
                #event_acc.Reload()

                # E.g. get wall clock, number of steps and value for a scalar 'Accuracy'
                #w_times, _, _ = zip(*event_acc.Scalars('train_loss'))
                #_, _, test_mse = zip(*event_acc.Scalars('test_loss'))

                flattened_dict = dict(utils.flattenize_dict(experiment_setup))
                # flattened_dict["model_name"] = flattened_dict["model_name"].split("_")[-1]
                flattened_dict["model_name"] = list(Path(experiment_path).parts)[-4]
                #flattened_dict["training_time"] = utils.get_training_duration(w_times[0], w_times[-1])

                # Get results
                exp_result = utils.get_performances(weather_data, experiment_path, experiment_setup,
                                                    filename='last.ckpt', best=False)

                # Get best val results
                list_models = os.listdir(os.path.join(experiment_path, "checkpoints"))
                best_model = [f for f in list_models if re.match(r'^epoch.*\.ckpt$', f)][0]

                exp_result_best_val = utils.get_performances(weather_data, experiment_path, experiment_setup,
                                                             filename=best_model, best=True)

                flattened_dict = {**flattened_dict, **exp_result, **exp_result_best_val}
                for key in flattened_dict:
                    results[key].append(flattened_dict[key])

                i = i + 1
                print(i)
                break
    df = pd.DataFrame(results)

    df.to_excel("results_matnet.xlsx")


if __name__ == "__main__":
    main()
