import argparse
import logging
import wandb
import pprint
import os
import pandas as pd
import lightning.pytorch as pl

from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_forecasting import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "config", help="Experiment configuration file in yaml format.")

    parser.add_argument(
        "-p",
        "--project",
        default="wne-masters-thesis-testing",
        help="W&B project name.")

    parser.add_argument(
        "-l",
        "--log-level",
        default=logging.INFO,
        type=int,
        help="Sets the log level.")

    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        type=int,
        help="Random seed for the training.")

    parser.add_argument(
        "-n",
        "--log-interval",
        default=100,
        type=int,
        help="Log every n steps."
    )

    parser.add_argument(
        '-v',
        '--val-check-interval',
        default=300,
        type=int,
        help="Run validation every n batches."
    )

    parser.add_argument(
        '-p',
        '--patience',
        default=8,
        type=int,
        help="Patience for early stopping."
    )

    parser.add_argument('--no-wandb', action='store_true',
                        help='Disables wandb, for testing.')

    return parser.parse_args()


def get_dataset(config, project):
    artifact_name = f"{project}/{config['data']['dataset']}"
    artifact = wandb.Api().artifact(artifact_name)
    base_path = artifact.download()
    logging.info(f"Artifacts downloaded to {base_path}")

    name = artifact.metadata['name']
    part_name = f"in-sample-{config['data']['sliding_window']}"
    data = pd.read_csv(os.path.join(
        base_path, name + '-' + part_name + '.csv'))
    logging.info(f"Using part: {part_name}")

    # TODO: Fix in dataset
    data['weekday'] = data['weekday'].astype('str')
    data['hour'] = data['hour'].astype('str')

    validation_part = config['data']['validation']
    logging.info(f"Using {validation_part} of in sample part for validation.")
    train_data = data.iloc[:int(len(data) * (1 - validation_part))]
    val_data = data.iloc[len(train_data) - config['past_window']:]
    logging.info(f"Trainin part size: {len(train_data)}")
    logging.info(
        f"Validation part size: {len(val_data)} "
        + f"({len(data) - len(train_data)} + {config['past_window']})")

    logging.info("Building time series dataset for training.")
    train = TimeSeriesDataSet(
        train_data,
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
        randomize_length=False
    )

    logging.info("Building time series dataset for validation.")
    val = TimeSeriesDataSet.from_dataset(
        train, val_data, stop_randomization=True)

    return train, val


def get_loss(config):
    loss_name = config['loss']['name']

    if loss_name == 'Quantile':
        return QuantileLoss(config['loss']['quantiles'])

    raise ValueError("Unknown loss")


def get_model(config, dataset, loss):
    model_name = config['model']['name']

    if model_name == 'TemporalFusionTransformer':
        return TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=config['model']['hidden_size'],
            dropout=config['model']['dropout'],
            attention_head_size=config['model']['attention_head_size'],
            hidden_continuous_size=config['model']['hidden_continuous_size'],
            learning_rate=config['model']['learning_rate'],
            share_single_variable_networks=False,
            loss=loss,
            logging_metrics=[MAE(), RMSE()]
        )

    raise ValueError("Unknown model")


def main():
    args = get_args()
    logging.basicConfig(level=args.log_level)
    pl.seed_everything(args.seed)

    run = wandb.init(
        project=args.project,
        config=args.config,
        job_type="train",
        mode="disabled" if args.no_wandb else "online"
    )
    config = run.config
    logging.info("Using experiment config:\n%s", pprint.pformat(config))

    # Get time series dataset
    train, valid = get_dataset(config, args.project)
    logging.info("Train dataset parameters:\n" +
                 f"{pprint.pformat(train.get_parameters())}")

    # Get loss
    loss = get_loss(config)
    logging.info(f"Using loss {loss}")

    # Get model
    model = get_model(config, train, loss)
    logging.info(f"Using model {config['model']['name']}")
    logging.info(f"{ModelSummary(model)}")
    logging.info(
        "Model hyperparameters:\n" +
        f"{pprint.pformat(model.hparams)}")

    # Checkpoint for saving the model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
        mode='min',
    )

    # Logger for W&B
    wandb_logger = WandbLogger(
        project=args.project,
        experiment=run,
        log_model="all") if not args.no_wandb else None

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience)

    batch_size = config['batch_size']
    logging.info(f"Training batch size {batch_size}.")

    epochs = config['max_epochs']
    logging.info(f"Training for {epochs} epochs.")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval
    )

    if epochs > 0:
        logging.info("Starting training:")
        trainer.fit(
            model,
            train_dataloaders=train.to_dataloader(
                batch_size=batch_size,
                num_workers=3,
            ),
            val_dataloaders=valid.to_dataloader(
                batch_size=batch_size, train=False, num_workers=3
            ))

    # Run validation with best model to log min val_loss
    # TODO: Maybe use different metric like min_val_loss ?
    ckpt_path = trainer.checkpoint_callback.best_model_path or None
    trainer.validate(model, dataloaders=valid.to_dataloader(
        batch_size=batch_size, train=False, num_workers=3),
        ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
