import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sccprnet.data import factories

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        factory,
        train_params,
        num_workers,
        bands,
        idx_test=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_params = train_params
        self.bands = bands
        self.idx_test = idx_test
        self.num_workers = num_workers

        factory_cls = factories.__dict__[factory.name]
        self.dataset_factory = factory_cls(bands=self.bands, **factory.params)

    def setup(self, stage=None):

        self.dataset_factory.setup()

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_factory.train(
                **self.train_params
            )
            self.val_dataset = self.dataset_factory.val()
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_factory.test(idx=self.idx_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # The patches live in LMDB and are copied to CUDA every iteration.
            # Keep workers alive and use page-locked host memory to overlap the
            # next batch preparation with the current GPU step.
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
