import logging

from omegaconf import errors
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from sccprnet import models
from sccprnet.data import DataModule
from sccprnet.callbacks import Backtracking
# from pytorch_lightning.callbacks import TQDMProgressBar

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    pl.seed_everything(cfg.seed)

    # Load datamodule
    datamodule = DataModule(**cfg.data.params)

    # Logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Callbacks
    callbacks = [
        cb.ModelCheckpoint(**cfg.checkpoint),
        # cb.ModelCheckpoint(**cfg.checkpoint_best),
        cb.LearningRateMonitor(),
        # cb.ProgressBar(),
        cb.TQDMProgressBar()
    ]
    try:
        logger.info("Loading backtracking config")
        callbacks.append(Backtracking(**cfg.model.backtracking))
        logger.info("Backtracking callback instantiated successfully")
    except (errors.ConfigAttributeError, TypeError):
        logger.info("Backtracking config not found")

    model_class = models.__dict__[cfg.model.class_name]
    # model = model_class.load_from_checkpoint(cfg.load_ckpt).to(device)
    model = model_class(**cfg.model.params)#.to(device)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="tb", name="", version=""
    )

    # Instantiate trainer
    # DDP adds process-synchronization overhead on a single GPU.  Retain the
    # explicit DDP strategy only when more than one visible device is used.
    strategy = (
        "auto"
        if cfg.trainer.params.devices == 1
        else "ddp_find_unused_parameters_true"
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="gpu",
        strategy=strategy,
        # devices=cfg.gpu_ids,
        **cfg.trainer.params,
    )

    # trainer.tune(model)

    # Print model info
    model.count_params()

    # Fit trainer
    
    trainer.fit(model, datamodule=datamodule,ckpt_path=cfg.ckpt_path)

    # Load the checkpoint selected by Lightning.  The previous implementation
    # looked for a hard-coded ``best`` directory even though checkpoints are
    # configured to be written to ``ckpts``.
    checkpoint_callback = trainer.checkpoint_callback
    path_best = checkpoint_callback.best_model_path
    if not path_best:
        path_best = checkpoint_callback.last_model_path

    if path_best:
        logger.info(f"Loading checkpoint for testing : {path_best}")
        checkpoint = torch.load(path_best, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        logger.warning("No checkpoint was saved; testing the final in-memory weights")

    # Report final metrics on the test split of the selected dataset.  The
    # legacy Tester is tied to hard-coded ICVL paths and is not valid when the
    # configured dataset is DCMall.
    trainer.test(model, datamodule=datamodule)
