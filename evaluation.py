import os
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src import utils

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
                               ]}


def main():
    i = 1
    for root, dirs, files in os.walk("./saved_models/MATNet", topdown=False):
        for name in dirs:
            if "version" in name:
                experiment_path = os.path.join(root, name)
                experiment_name = list(Path(experiment_path).parts)[-3]
                experiment_setup = utils.filename2setup(experiment_name)

                flattened_dict = dict(utils.flattenize_dict(experiment_setup))
                # flattened_dict["model_name"] = flattened_dict["model_name"].split("_")[-1]
                flattened_dict["model_name"] = list(Path(experiment_path).parts)[-4]

                # Get results
                exp_result = utils.get_performances(experiment_path, experiment_setup,
                                                    filename='last.ckpt', best=False)

                flattened_dict = {**flattened_dict, **exp_result}
                for key in flattened_dict:
                    results[key].append(flattened_dict[key])

                i = i + 1
                print(i)
                break
    df = pd.DataFrame(results)

    df.to_excel("results_matnet.xlsx")


if __name__ == "__main__":
    main()
