import argparse
import logging
import wandb
import pprint
import os
import tempfile
import torch
import lightning.pytorch as pl
import pandas as pd

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
        test_data = pd.concat(
            [valid_data[-config['past_window']:], out_of_sample])
        test = build_time_series_dataset(run.config, test_data)
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
