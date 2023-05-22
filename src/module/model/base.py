import traceback
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import LightningModule
from timm.scheduler import CosineLRScheduler
from torch import nn

# model
from src.ecapa_tdnn.model import ECAPA_TDNN
from src.ecapa_tdnn.preprocess import Wave2MelSpecPreprocess
from src.utils.augment import FbankMaskAug

# loss and metrics
from src.criteria.utils import get_loss
from src.metrics.metrics import accuracy, ComputeErrorRates, ComputeMinDcf
from src.metrics.utils import tuneThresholdfromScore

# utils
from src.utils.audio import load_wave
from src.utils.logger import get_logger

logger = get_logger(debug=True)

class EcapaTdnnModelModule(LightningModule):
    def __init__(self, cfg, train_audio_file_list=[]):
        super(EcapaTdnnModelModule, self).__init__()

        self.model = ECAPA_TDNN(
            cfg.model.ecapa_tdnn.channel_size,
            cfg.model.ecapa_tdnn.hidden_size
        )
        
        self._preprocesser = Wave2MelSpecPreprocess(
            sample_rate=cfg.dataset.audio.sample_rate,
            n_fft=cfg.model.preprocess.n_fft,
            win_length=cfg.model.preprocess.win_length,
            hop_length=cfg.model.preprocess.hop_length,
            n_mels=cfg.model.preprocess.n_mels,
            f_min=cfg.model.preprocess.f_min,
            f_max=cfg.model.preprocess.f_max
        )
        
        self._fbank_mask_aug = FbankMaskAug(
            freq_mask_width=cfg.dataset.augment.freq_mask_width,
            time_mask_width=cfg.dataset.augment.time_mask_width
        )
        
        self._loss = get_loss(
            cfg.ml.loss.type,
            spk_index_info_json_path=cfg.dataset.train.spk_index_info_json_path,
            use_ce_weight=cfg.ml.loss.use_ce_weight,
            audio_file_list=train_audio_file_list,
            hidden_size=cfg.model.ecapa_tdnn.hidden_size,
            aam_margin=cfg.ml.loss.aam.margin,
            aam_scale=cfg.ml.loss.aam.scale
        )
        
        self.sample_rate = cfg.dataset.audio.sample_rate
        self.waveform_length = cfg.dataset.audio.waveform_length
        self.hop_length = cfg.model.preprocess.hop_length
        self._batch_size = cfg.ml.batch_size
        
        self._learning_rate = cfg.ml.learning_rate
        self._optimizer_type = cfg.ml.optimizer.type
        self._optimizer_weight_decay = cfg.ml.optimizer.weight_decay
        self._optimizer_eps = cfg.ml.optimizer.opt_eps
        self._adan_params = cfg.ml.optimizer.adan
        

        self._sh_params = cfg.ml.scheduler
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        audio, labels, _ = batch
        audio = self._preprocesser(audio)
        audio = self._fbank_mask_aug(audio)
        output = self.forward(audio)
        
        loss, output = self._loss(output, labels)

        prec = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', prec, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self._embeddings = {}
        self._scores = []
        self._labels = []
    
    def validation_step(self, batch, batch_idx):
        _, labels, audio_files = batch
        for label, audio_file in zip(labels, audio_files):
            label = int(label.cpu().item())
            audio, _ = load_wave(audio_file, sample_rate=self.sample_rate, is_torch=False, mono=True)
            data_1 = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            max_audio = 300 * self.hop_length  + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = np.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
            startframe = list(set(startframe))
            for asf in startframe:
                feats.append(audio[int(asf):int(asf + max_audio)])
            feats = np.stack(feats, axis=0)
            data_2 = torch.FloatTensor(feats).to(self.device)
            with torch.no_grad():
                emb_1 = self._preprocesser(data_1)
                emb_1 = self.forward(emb_1)
                emb_1 = F.normalize(emb_1, p=2, dim=1)
                emb_2 = self._preprocesser(data_2)
                emb_2 = self.forward(emb_2)
                emb_2 = F.normalize(emb_2, p=2, dim=1)
            if label not in self._embeddings:
                self._embeddings[label] = []
            self._embeddings[label].append([emb_1.detach().cpu(), emb_2.detach().cpu()])
        
    def on_validation_epoch_end(self) -> None:
        for k, v in self._embeddings.items():
            for i, (emb_11, emb_12) in enumerate(v):
                if i == 0:
                    emb_21, emb_22 = emb_11, emb_12
                    continue
                score_1 = torch.mean(torch.matmul(emb_11, emb_21.T))
                score_2 = torch.mean(torch.matmul(emb_12, emb_22.T))
                score = (score_1 + score_2) / 2
                score = score.detach().cpu().numpy()
                self._scores.append(score)
                self._labels.append(k)
                emb_21, emb_22 = emb_11, emb_12
        try:
            eer = tuneThresholdfromScore(self._scores, self._labels, [1, 0.1])[1]
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            eer = 1.0
        fnrs, fprs, thresholds = ComputeErrorRates(self._scores, self._labels)
        minDCF, _  = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        
        self.log('val_eer', eer, on_step=False, on_epoch=True, logger=True)
        self.log('val_minDCF', minDCF, on_step=False,on_epoch=True, logger=True)
        
        # clear
        self._scores.clear()
        self._labels.clear()
        self._embeddings.clear()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self._optimizer_weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        if self._optimizer_type == 'adan':
            from adan import Adan
            self.optimizer = Adan(
                optimizer_grouped_parameters,
                lr=self._learning_rate,
                betas=self._adan_params.opt_betas,
                eps=self._optimizer_eps,
                max_grad_norm=self._adan_params.max_grad_norm,
                no_prox=self._adan_params.no_prox,
                fused=self._adan_params.fused
            )
        elif self._optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self._learning_rate,
                eps=self._optimizer_eps,
            )
        else:
            raise NotImplementedError()
        
        self.scheduler  = CosineLRScheduler(
            self.optimizer,
            t_initial=self._sh_params.t_initial,
            cycle_mul=self._sh_params.t_mul,
            cycle_decay=self._sh_params.decay_rate,
            warmup_t=self._sh_params.warm_up_t,
            warmup_lr_init=self._sh_params.warm_up_init,
            warmup_prefix=self._sh_params.warmup_prefix,
        )
        
        return [self.optimizer], [
            {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self._sh_params.monitor,
            }
        ]
    
    def lr_scheduler_step(self, scheduler, metric):
        # timm's scheduler need the epoch value
        scheduler.step(epoch=self.current_epoch)