import os

import pandas as pd
import wandb
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet


def get_dataset_from_wandb(run, window=None):
    artifact_name = f"{run.project}/{run.config['data']['dataset']}"
    artifact = wandb.Api().artifact(artifact_name)
    base_path = artifact.download()

    name = artifact.metadata['name']
    in_sample_name = f"in-sample-{window or run.config['data']['sliding_window']}"
    in_sample_data = pd.read_csv(os.path.join(
        base_path, name + '-' + in_sample_name + '.csv'))
    out_of_sample_name = f"out-of-sample-{window or run.config['data']['sliding_window']}"
    out_of_sample_data = pd.read_csv(os.path.join(
        base_path, name + '-' + out_of_sample_name + '.csv'))

    return in_sample_data, out_of_sample_data

def get_train_validation_split(config, in_sample_data):
    validation_part = config['data']['validation']
    train_data = in_sample_data.iloc[:int(len(in_sample_data) * (1 - validation_part))]
    val_data = in_sample_data.iloc[len(train_data) - config['past_window']:]

    return train_data, val_data


def build_time_series_dataset(config, data):
    data = data.copy()
    data['weekday'] = data['weekday'].astype('str')
    data['hour'] = data['hour'].astype('str')

    time_series_dataset = TimeSeriesDataSet(
        data,
        time_idx=config['data']['fields']['time_index'],
        target=config['data']['fields']['target'],
        group_ids=config['data']['fields']['group_ids'],
        min_encoder_length=config['past_window'],
        max_encoder_length=config['past_window'],
        min_prediction_length=config['future_window'],
        max_prediction_length=config['future_window'],
        static_reals=config['data']['fields']['static_real'],
        static_categoricals=config['data']['fields']['static_cat'],
        time_varying_known_reals=config['data']['fields']['dynamic_known_real'],
        time_varying_known_categoricals=config['data']['fields']['dynamic_known_cat'],
        time_varying_unknown_reals=config['data']['fields']['dynamic_unknown_real'],
        time_varying_unknown_categoricals=config['data']['fields']['dynamic_unknown_cat'],
        randomize_length=False,
    )

    return time_series_dataset