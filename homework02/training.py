import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def setup_trainer(
    warmup_epochs,
    decay_epochs,
    accumulate_batches,
    tb_logs,
    tb_model_name,
    checkpoint_path,
    full_precision
):
    logger = pl.loggers.TensorBoardLogger(tb_logs, name=tb_model_name)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-3, patience=20, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, monitor="val_loss", mode="min"
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_batches,
        gpus=1,
        precision= 32 if full_precision else 16,
        logger=logger,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        max_epochs= warmup_epochs + decay_epochs,
        weights_summary=None,
        log_gpu_memory="all",
    )

    return trainer