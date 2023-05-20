import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

class PreEmphasis(torch.nn.Module):
    """
    高周波を強調し、低周波のamplitudeを小さくする
    y_t = x_t - conf * x_{t-1}
    """
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class Wave2MelSpecPreprocess(nn.Module):
    """波形データに対する前処理
    """
    def __init__(
        self, 
        sample_rate=16000, 
        n_fft=512, 
        win_length=400, 
        hop_length=160, 
        f_min=20,
        f_max=7600,
        n_mels=80
    ):
        super(Wave2MelSpecPreprocess, self).__init__()
        
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, \
                                                 f_min = f_min, f_max = f_max, window_fn=torch.hamming_window, n_mels=n_mels),
            )
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x (torch.tensor): 音声波形データ (batch_size, frame_size)

        Returns:
            torch.tensor: 前処理の結果 (batch_size, channel_size:n_mels, 変換後のframe_size)
        """
        with torch.no_grad():
            x = self.torchfbank(x)
            x = x - torch.mean(x, dim=-1, keepdim=True)
        return x