import numpy as np
import torch
import gc
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
import random
from scipy import signal

from src.utils.audio import load_wave

from audiomentations import Compose, TimeStretch
from src.dataset.augments.volume import VolumeAugment

class VoiceDataset(Dataset):
    def __init__(self, 
                audio_file_list: List,
                sample_rate: int = 16000,
                waveform_length: int = 32240, # 200 frame * 160 hop_length + 240
                is_aug:bool = False,
                use_noise=True,
                musan_path: str = "/data/musan",
                rir_path: str = "/data/riris_noises",
                time_stretch_params: Tuple[float, float, float] = (0.8, 1.2, 0.5),
                volume_mul_params: List = [0,1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                volume_aug_rate: float = 0.8,
                is_audio_file_only: bool = False,
        ):
        """音声とラベルと返すデータセット
        速度変化、ノイズ追加、リバーブ追加のデータ拡張

        Args:
            audio_file_list (List):[(label, audio_file)]のデータセットリスト
            sample_rate (int, optional): サンプリングレート. Defaults to 16000.
            waveform_length (int, optional): waveformの長さ. Defaults to 32240.
            is_aub (bool): データ拡張をするか. Defaults to False.
            musan_path (str, optional): ノイズデータ(musan)のディレクトリ. Defaults to "/data/musan".
            rir_path (str, optional): リバーブ作成用音のディレクトリ. Defaults to "/data/riris_noises".
            time_stretch_params (Tuple[float, float, float], optional): 時間データ拡張のパラメータ [最小掛け率, 最大掛け率, 実行確率]. Defaults to (0.8, 1.2, 0.5).
        """
        self.audio_file_list = audio_file_list
        self.sample_rate = sample_rate
        self.waveform_length = waveform_length
        self.is_aug = is_aug
        self.is_audio_file_only = is_audio_file_only
        self.use_noise = use_noise
        
        self.__aug_setup(
            musan_path, 
            rir_path, 
            time_stretch_params,
            volume_mul_params,
            volume_aug_rate
        )
    
    def __len__(self):
        return len(self.audio_file_list)
    
    def __getitem__(self, idx):
        label, audio_file = self.audio_file_list[idx]
        if self.is_audio_file_only:
            return 0, label, audio_file
        
        add_index = 1
        while True:
            try:
                label, audio_file = self.audio_file_list[idx]
                waveform, _ = load_wave(audio_file, sample_rate=self.sample_rate, is_torch=False, mono=True)
                if np.abs(waveform).max() <= 0:
                    break
                if self.is_aug:
                    waveform = self.waveform_aug(waveform, sample_rate=self.sample_rate)
                    waveform = self.volume_aug(waveform)
                break
            except Exception as e:
                print(e)
                idx = int((idx + add_index) % len(self.audio_file_list))
                add_index += 1
                continue
        
        # waveform aug (time stretch, volume)

        # waveformをlengthで切り出す
        if waveform.shape[0] <= self.waveform_length:
            shortage = self.waveform_length - waveform.shape[0]
            waveform = np.pad(waveform, (0, shortage), "wrap")
        start_frame = np.int64(torch.rand(1).item() * (waveform.shape[0] - self.waveform_length))
        waveform = waveform[start_frame:start_frame + self.waveform_length]
        waveform = np.stack([waveform], axis=0)
        
        # データ拡張
        if self.is_aug:
            waveform = self._augment(waveform)
        return torch.FloatTensor(waveform[0]), label, audio_file
    
    def __aug_setup(
        self, 
        musan_path, 
        rir_path, 
        time_stretch_params,
        volume_mul_params,
        volume_aug_rate,
    ):
        if not self.is_aug:
            return
        self.waveform_aug = Compose([
            TimeStretch(min_rate=time_stretch_params[0], max_rate=time_stretch_params[1], p=time_stretch_params[2])
        ])
        
        self.volume_aug = VolumeAugment(
            volume_mul_params,
            volume_aug_rate
        )
        
        if self.use_noise:
            # Musan: Load and configure augmentation files
            ################################
            self.noisetypes = ['noise','speech','music']
            self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
            self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
            
            ################################
            self.noiselist = {}
            augment_files = sorted(list(Path(musan_path).glob('*/*/*.wav')))
            assert len(augment_files) > 0, f"No musan files found in {musan_path}"
            for file in augment_files:
                noise_type = file.parent.parent.stem
                if noise_type not in self.noiselist:
                    self.noiselist[noise_type] = []
                self.noiselist[noise_type].append(str(file))

        # Rir files
        self.rir_filies = sorted(list(Path(rir_path).glob('*/*/*.wav')))
        assert len(self.rir_filies) > 0, f"No rir files found in {rir_path}"
        
    def _augment(self, waveform):
        if self.use_noise:
            augtype = torch.randint(0, 6, (1, 1)).item()
            if augtype == 0:
                pass # original
            elif augtype == 1:
                waveform = self.__add_rev(waveform)
            elif augtype == 2:
                waveform = self.__add_noise(waveform, 'speech')
            elif augtype == 3:
                waveform = self.__add_noise(waveform, 'music')
            elif augtype == 4:
                waveform = self.__add_noise(waveform, 'noise')
            elif augtype == 5:
                waveform = self.__add_noise(waveform,'speech')
                waveform = self.__add_noise(waveform, 'music')
            else:
                raise ValueError(f"Unknown noise type: {augtype}")
        else:
            augtype = torch.randint(0, 3, (1, 1)).item()
            if augtype in  [0, 1]:
                pass # original
            elif augtype in [2]:
                waveform = self.__add_rev(waveform)
            else:
                raise ValueError(f"Unknown noise type: {augtype}")
        return waveform
    
    def __add_rev(self, waveform):
        rir_file = random.choice(self.rir_filies)
        rir, sr = load_wave(rir_file, sample_rate=self.sample_rate, is_torch=False, mono=True)
        rir = np.expand_dims(rir.astype(np.float64), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        rir = signal.convolve(waveform, rir, mode="full")[:, :self.waveform_length]
        return rir
    
    def __add_noise(self, waveform, noisecat):
        clean_db = 10 * np.log10(np.mean(waveform ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = load_wave(noise, sample_rate=self.sample_rate, is_torch=False, mono=True)
            if noiseaudio.shape[0] <= self.waveform_length:
                shortage = self.waveform_length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), "wrap")
            start_frame = np.int64(torch.rand(1).item() * (noiseaudio.shape[0] - self.waveform_length))
            noiseaudio = noiseaudio[start_frame:start_frame + self.waveform_length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + waveform


if __name__ == "__main__":
    from src.utils.audio import save_wave
    dataset = VoiceDataset(
        [(0, "tests/__example/test.wav")],
        is_aug=True
    )
    
    Path("./results/test_audio").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        waveform, label, audio_file = dataset[0]
        print(waveform.shape, label, audio_file)
        save_wave(waveform, f"./results/test_audio/dataset_sample_{i:05d}.wav")