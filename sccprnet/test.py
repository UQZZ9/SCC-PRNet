import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from sccprnet import models
from sccprnet.data import DataModule

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device in use : {device}")

    # Fix seed for reproducibility
    logger.info(f"Using random seed {cfg.seed}")
    seed_everything(cfg.seed)

    # Load requested dataloader
    datamodule = DataModule(idx_test=cfg.test.idx_test, **cfg.data.params)
    datamodule.setup(stage="test")

    model_class = models.__dict__[cfg.model.class_name]
    model = model_class.load_from_checkpoint(cfg.ckpt_path)

    model = model.to(device)

    # Evaluate the split selected by the configured DataModule (DCMall for
    # ``data=dcmall``).  The legacy Tester reads an unrelated, hard-coded ICVL
    # directory and therefore does not measure the requested DCMall test set.
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=datamodule)
