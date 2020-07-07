import os

import click

from torch.utils.data import DataLoader

from data_processing import load_data, prepare_data
from model import MyNet, init_fn
from model_checkpoint import MyNetCheckpoint
from training import setup_trainer


@click.command()
@click.option("--batch-size", default=512, show_default=True)
@click.option("--accumulate-batches", default=1)
@click.option(
    "--val-size",
    default=0.1,
    show_default=True,
    help="Share of train set (size is 10^5) that should be used for validation.",
)
@click.option("--checkpoint-segments", default=1, show_default=True)
@click.option("--full-precision", is_flag=True, help='Train the model with 32-bit floats [default 16-bit]')
@click.option("--warmup-epochs", default=5, show_default=True)
@click.option("--decay-epochs", default=55, show_default=True)
@click.option("--initial-channel-size", default=32, show_default=True)
@click.option(
    "--tb-logs",
    default="./tb_logs",
    type=click.Path(exists=True),
    show_default=True,
    help="Directory where Tensorboard logs will be written.",
)
@click.option("--tb-model-name", default="my_model", help="Model name for Tensorboard")
@click.option(
    "--checkpoint-path",
    default="./checkpoints",
    type=click.Path(exists=True),
    show_default=True,
    help="Directory where model checkpoints will be saved.",
)
def main_script(
    batch_size,
    accumulate_batches,
    val_size,
    checkpoint_segments,
    full_precision,
    warmup_epochs,
    decay_epochs,
    initial_channel_size,
    tb_logs,
    tb_model_name,
    checkpoint_path,
):

    # Load data if necessary
    load_data()

    # Prepare data
    train_set, val_set, test_set = prepare_data(val_size)

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=8, pin_memory=True
    )
    
    # Create and initialize model
    if checkpoint_segments == 1:
        model = MyNet(
            int(len(train_loader) / accumulate_batches)+1,
            warmup_epochs,
            decay_epochs,
            initial_channel_size,
        )
    else:
        model = MyNetCheckpoint(
            int(len(train_loader) / accumulate_batches)+1,
            warmup_epochs,
            decay_epochs,
            initial_channel_size,
            checkpoint_segments,
        )

    model = model.apply(init_fn)
    
    # Prepare the trainer
    trainer = setup_trainer(
        warmup_epochs,
        decay_epochs,
        accumulate_batches,
        tb_logs,
        tb_model_name,
        checkpoint_path,
        full_precision
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main_script()
