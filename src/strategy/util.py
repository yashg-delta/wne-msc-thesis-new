import wandb
import os
import torch
import pandas as pd
import numpy as np
from numba import jit
from numba import int32, float64, optional


def get_sweep_data_windows(sweep_id):
    """Get all data windows evaluated during sweep moving window eval."""
    sweep = wandb.Api().sweep(sweep_id)
    sweep_dataset = sweep.config[
        'parameters']['data']['parameters']['dataset']['value']
    sliding_window_min = sweep.config[
        'parameters']['data']['parameters']['sliding_window']['min']
    slidinw_window_max = sweep.config[
        'parameters']['data']['parameters']['sliding_window']['max']

    return get_data_windows(
        sweep.project,
        sweep_dataset,
        min_window=sliding_window_min,
        max_window=slidinw_window_max)


def get_data_windows(project, dataset_name, min_window=0, max_window=5):
    artifact_name = f"{project}/{dataset_name}"
    artifact = wandb.Api().artifact(artifact_name)
    base_path = artifact.download()
    name = artifact.metadata['name']

    result = []
    for i in range(min_window, max_window+1):
        in_sample_name =\
            f"in-sample-{i}"
        in_sample_data = pd.read_csv(os.path.join(
            base_path, name + '-' + in_sample_name + '.csv'))
        out_of_sample_name =\
            f"out-of-sample-{i}"
        out_of_sample_data = pd.read_csv(os.path.join(
            base_path, name + '-' + out_of_sample_name + '.csv'))
        result.append((in_sample_data, out_of_sample_data))

    return result


def get_sweep_window_predictions(sweep_id, part):
    result = []
    for run in wandb.Api().sweep(sweep_id).runs:
        window_num = run.config['data']['sliding_window']

        window_prediction = list(
            filter(lambda x: (
                x.type == 'prediction'
                and x.name.startswith(f'prediction-{part}')),
                run.logged_artifacts()))

        assert len(window_prediction) == 1
        window_prediction = window_prediction[0]

        artifact_path = window_prediction.download()
        index = torch.load(os.path.join(
            artifact_path, 'index.pt'), map_location=torch.device('cpu'))
        preds = torch.load(os.path.join(
            artifact_path, 'predictions.pt'), map_location=torch.device('cpu'))

        result.append((window_num, index, preds.numpy()))

    result = sorted(result, key=lambda x: x[0])
    return result


def get_predictions_dataframe(*window_predictions):
    result = []
    for _, idx, preds in window_predictions:
        df = pd.DataFrame(idx)
        df['prediction'] = list(preds)
        result.append(df)

    result = pd.concat(result).sort_values(by='time_index')

    assert 'time_index' in result.columns
    assert 'group_id' in result.columns
    assert 'prediction' in result.columns

    return result


@jit((float64[:], int32, int32), nopython=True)
def rsi_obos(rsi_arr, oversold, overbought):
    moves = np.zeros(rsi_arr.size, dtype=np.int32)
    for i in range(1, rsi_arr.size):
        moves[i] = 1 if rsi_arr[i - 1] < oversold and rsi_arr[i] > oversold \
            else 0 if rsi_arr[i - 1] > overbought and rsi_arr[i] < overbought \
            else moves[i - 1]
    return moves


# @jit((
#     float64[:],
#     optional(int32),
#     optional(int32),
#     optional(int32),
#     optional(int32)), nopython=True)
# def quantile_model_pos(
#         preds,
#         enter_long,
#         exit_long,
#         enter_short,
#         exit_short):
#     return None
#     # moves = np.zeros(preds.size, dtype=np.int32)
#     # for i in range(1, preds.size):


# if __name__ == '__main__':
#     quantile_model_pos(np.array([0.2, 0.1, 0.3]), 2, 5, None, None)
