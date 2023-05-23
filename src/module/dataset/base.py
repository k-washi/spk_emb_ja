import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from typing import List, Optional

# dataset lib
from src.dataset.utils import collect_dataset_from_dataset_list
from src.dataset.dataset import VoiceDataset

# utils
from src.utils.ml import seed_worker
from src.utils.logger import get_logger

logger = get_logger(debug=True)


class VoiceDataModule(LightningDataModule):
    def __init__(self, cfg, train_audio_list, test_audio_list) -> None:
        super().__init__()
        
        self._sample_rate = cfg.dataset.audio.sample_rate
        self._waveform_length = cfg.dataset.audio.waveform_length
        
        self._use_noise = cfg.dataset.augment.use_noise
        self._musan_path = cfg.dataset.augment.musan_dir_path
        self._rir_path = cfg.dataset.augment.rir_dir_path
        self._time_stretch_aug_params = cfg.dataset.augment.time_stretch_params
        
        self._batch_size = cfg.ml.batch_size
        self._num_workers = cfg.ml.dataloader.num_workers
        
        self._train_audio_list = train_audio_list
        self._val_audio_list = test_audio_list
        self._test_audio_list = test_audio_list
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = VoiceDataset(
            self._train_audio_list,
            sample_rate=self._sample_rate,
            waveform_length=self._waveform_length,
            is_aug=True,
            use_noise=self._use_noise,
            musan_path=self._musan_path,
            rir_path=self._rir_path,
            time_stretch_params=self._time_stretch_aug_params,
            is_audio_file_only=False
        )
        
        logger.info(f"Train Dataset Size: {len(dataset)}")
        
        #@reference: https://pytorch.org/docs/stable/notes/randomness.html
        # g = torch.Generator()
        # g.manual_seed(0)
    
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = VoiceDataset(
            self._val_audio_list,
            sample_rate=self._sample_rate,
            waveform_length=self._waveform_length,
            is_aug=False,
            use_noise=self._use_noise,
            is_audio_file_only=True
        )
        
        logger.info(f"Val Dataset Size: {len(dataset)}")
        
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=seed_worker
        )
    def test_dataloader(self) -> EVAL_DATALOADERS:
        dataset = VoiceDataset(
            self._test_audio_list,
            sample_rate=self._sample_rate,
            waveform_length=self._waveform_length,
            is_aug=False,
            use_noise=self._use_noise,
            is_audio_file_only=True
        )
        
        logger.info(f"Test Dataset Size: {len(dataset)}")
        
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=seed_worker
        )