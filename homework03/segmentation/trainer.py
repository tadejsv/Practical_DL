import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger


def setup_trainer(model_name, max_epochs):

    # Loggers
    tb_logger = TensorBoardLogger('tb_logs', name=model_name)
    wandb_logger = WandbLogger(
        name=model_name, project='practical-dl-segmentation')

    # Other callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-3, patience=20, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints', monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateLogger()

    # Finally, set up trainer
    trainer = pl.Trainer(
        #        gpus=1,
        precision=32,
        logger=[tb_logger, wandb_logger],
        callbacks=[lr_logger],
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        weights_summary=None,
        log_gpu_memory="all",
    )

    return trainer
