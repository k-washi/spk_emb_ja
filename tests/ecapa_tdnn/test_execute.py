import torch
from src.ecapa_tdnn import ECAPA_TDNN, Wave2MelSpecPreprocess
from src.utils.audio import load_wave
from src.utils.augment import FbankMaskAug


TEST_AUDIO_FILE = "tests/__example/test.wav"
SAMPLE_RATE = 16000
HOP_LENGTH = 160

HIDDEN_SIZE = 128

def test_ecapa_tdnn_execute():
    preprocesser = Wave2MelSpecPreprocess(
        sample_rate=SAMPLE_RATE,
        n_fft=512, 
        win_length=400, 
        hop_length=HOP_LENGTH, 
        f_min=20,
        f_max=7600,
        n_mels=80
    )
    
    model = ECAPA_TDNN(
        channel_size=1024,
        hidden_size=HIDDEN_SIZE
    )
    
    fbank_aug = FbankMaskAug()
    
    wave_data, _ = load_wave(TEST_AUDIO_FILE, sample_rate=SAMPLE_RATE, is_torch=True, mono=False)
    for time_index_num in [9, 49, 99, 380]:
        x = wave_data[:, :int(HOP_LENGTH*time_index_num)]
        x = preprocesser(x)
        x = fbank_aug(x)
        _, _, time_index = x.size()
        assert time_index == time_index_num+1, f"preprocess error"
        
        model.eval()
        with torch.no_grad():
            y = model.vecterize(x)
            _, hs = y.size()
            assert hs == HIDDEN_SIZE, f"vectorize error"
            y = model(x)
            _, hs = y.size()
            assert hs == HIDDEN_SIZE, f"forward error"