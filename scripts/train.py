import argparse
import logging
import wandb
import pprint
import os
import tempfile
import torch
import lightning.pytorch as pl

from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from ml.loss import get_loss
from ml.model import get_model
from ml.data import (
    get_dataset_from_wandb,
    get_train_validation_split,
    build_time_series_dataset
)


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
        '-t',
        '--patience',
        default=5,
        type=int,
        help="Patience for early stopping."
    )

    parser.add_argument('--no-wandb', action='store_true',
                        help='Disables wandb, for testing.')

    parser.add_argument('--store-predictions', action='store_true',
                        help='Whether to store predictions of the best run.')

    return parser.parse_args()


# def get_dataset(config, project):
#     artifact_name = f"{project}/{config['data']['dataset']}"
#     artifact = wandb.Api().artifact(artifact_name)
#     base_path = artifact.download()
#     logging.info(f"Artifacts downloaded to {base_path}")

#     name = artifact.metadata['name']
#     part_name = f"in-sample-{config['data']['sliding_window']}"
#     data = pd.read_csv(os.path.join(
#         base_path, name + '-' + part_name + '.csv'))
#     logging.info(f"Using part: {part_name}")

#     # TODO: Fix in dataset
#     data['weekday'] = data['weekday'].astype('str')
#     data['hour'] = data['hour'].astype('str')

#     validation_part = config['data']['validation']
#     logging.info(f"Using {validation_part} of in sample part for validation.")
#     train_data = data.iloc[:int(len(data) * (1 - validation_part))]
#     val_data = data.iloc[len(train_data) - config['past_window']:]
#     logging.info(f"Trainin part size: {len(train_data)}")
#     logging.info(
#         f"Validation part size: {len(val_data)} "
#         + f"({len(data) - len(train_data)} + {config['past_window']})")

#     logging.info("Building time series dataset for training.")
#     train = TimeSeriesDataSet(
#         train_data,
#         time_idx=config['data']['fields']['time_index'],
#         target=config['data']['fields']['target'],
#         group_ids=config['data']['fields']['group_ids'],
#         min_encoder_length=config['past_window'],
#         max_encoder_length=config['past_window'],
#         min_prediction_length=config['future_window'],
#         max_prediction_length=config['future_window'],
#         static_reals=config['data']['fields']['static_real'],
#         static_categoricals=config['data']['fields']['static_cat'],
#         time_varying_known_reals=config['data']['fields'][
#             'dynamic_known_real'],
#         time_varying_known_categoricals=config['data']['fields'][
#             'dynamic_known_cat'],
#         time_varying_unknown_reals=config['data']['fields'][
#             'dynamic_unknown_real'],
#         time_varying_unknown_categoricals=config['data']['fields'][
#             'dynamic_unknown_cat'],
#         randomize_length=False
#     )

#     logging.info("Building time series dataset for validation.")
#     val = TimeSeriesDataSet.from_dataset(
#         train, val_data, stop_randomization=True)

#     return train, val


# def get_loss(config):
#     loss_name = config['loss']['name']

#     if loss_name == 'Quantile':
#         return QuantileLoss(config['loss']['quantiles'])

#     if loss_name == 'GMADL':
#         return GMADL(
#             a=config['loss']['a'],
#             b=config['loss']['b']
#         )

#     raise ValueError("Unknown loss")


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
    in_sample, out_of_sample = get_dataset_from_wandb(run)
    train_data, valid_data = get_train_validation_split(run.config, in_sample)
    train = build_time_series_dataset(run.config, train_data)
    valid = build_time_series_dataset(run.config, valid_data)
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

    if not args.no_wandb and args.store_predictions:
        logging.info("Computing and storing predictions of best model.")
        test = build_time_series_dataset(run.config, out_of_sample)
        test_preds = model.__class__.load_from_checkpoint(ckpt_path).predict(
            test.to_dataloader(train=False, batch_size=batch_size),
            mode="raw",
            return_index=True,
            trainer_kwargs={
                'logger': False
            })

        with tempfile.TemporaryDirectory() as tempdir:
            for key, value in {
                'index': test_preds.index,
                'predictions': test_preds.output.prediction
            }.items():
                torch.save(value, os.path.join(tempdir, key + ".pt"))
            pred_artifact = wandb.Artifact(
                f"prediction-{run.id}", type='prediction')
            pred_artifact.add_dir(tempdir)
            run.log_artifact(pred_artifact)

    # Clean up models that do not have best/latest tags, to save space on wandb
    for artifact in wandb.Api().run(run.path).logged_artifacts():
        if artifact.type == "model" and not artifact.aliases:
            logging.info(f"Deleting artifact {artifact.name}")
            artifact.delete()


if __name__ == '__main__':
    main()
