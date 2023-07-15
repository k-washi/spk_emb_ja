import torch
import numpy as np

class VolumeAugment(torch.nn.Module):
    def __init__(
        self, 
        volume_mul_params:list=[0.25, 0.5, 0.75, 0.95], 
        volume_aug_rate:float=0.8
    ) -> None:
        super().__init__()
        
        self.volume_mul_params = volume_mul_params
        self.volume_aug_rate = volume_aug_rate
        
    def forward(self, x:np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (torch.Tensor): waveform (audio length)

        Returns:
            torch.Tensor: augmented waveform
        """
        
        if torch.rand((1)).item() > self.volume_aug_rate:
            return x
        
        mul_param_index = torch.randint(0, len(self.volume_mul_params), size=(1,)).item()
        mul_param = self.volume_mul_params[mul_param_index]
        volume_max = np.abs(x).max()
        assert volume_max > 0, "volume_max should be greater than 0"
        x = x / np.abs(x).max() * mul_param
        return x
        
        
        