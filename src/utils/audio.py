import torch
import torchaudio
import torchaudio.transforms as at

def load_wave(wave_file_path:str, sample_rate:int=-1, is_torch:bool=True, mono:bool=False):
    """_summary_

    Args:
        wave_file_path (str): file path
        sample_rate (int, optional): if -1 return original sample rate. Defaults to -1.
        is_torch (bool, optional): return torch.Tensor or np.ndarray. Defaults to True.
        mono (bool, optional):
            True: return [wave]
            False: return [channel, wave]. 
            Defaults to False.

    Returns:
        wave torch.Tensor or np.ndarray return 
        sample_rate (int)
    """
    
    
    wave, sr = torchaudio.load(wave_file_path)
    if mono:
        wave = wave[0]
    if sample_rate > 0 and sample_rate != sr:
        wave = torchaudio.transforms.Resample(sr, sample_rate)(wave)
    else:
        sample_rate=sr
    if not is_torch:
        wave = wave.cpu().detach().numpy().copy()
    return wave, sample_rate

def save_wave(wave, output_path, sample_rate:int=16000):
    """save wave
    """
    if not isinstance(wave, torch.Tensor):
        wave = torch.from_numpy(wave)

    if wave.dim() == 1: wave = wave.unsqueeze(0)
    torchaudio.save(filepath=str(output_path), src=wave.to(torch.float32), sample_rate=sample_rate)