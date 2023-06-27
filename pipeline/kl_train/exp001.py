import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.module.dataset.base import VoiceDataModule
from src.module.model.kl_norm_dist import EcapaTdnnKLModelModule

from src.dataset.utils import collect_dataset_from_dataset_list

from src.utils.logger import get_logger
logger = get_logger(debug=True)

##########
# PARAMS #
##########

EXP_ID = "0023"
LOG_SAVE_DIR = f"logs/kl_{EXP_ID}"
MODEL_SAVE_DIR = f"checkpoints/kl_{EXP_ID}"


#TRAIN_DATASET_LIST =  ["/data/jvs_vc"] 
TRAIN_DATASET_LIST =  ["/data/jvs_vc", "/data/common_voice", "/data/lecture_vc", "/data/vc_dataset"]

FAST_DEV_RUN = False # 確認用の実行を行うか

# TRAIN PARAMS
NUM_EPOCHS = 1000
BATCH_SIZE = 128
SCHEDULER_T_INITIAL = 10

AUGMENT_TIME_STRETCH_PARAMS = [0.95, 1.05, 0.5]

# MODEL PARAMS
HIDDEN_SIZE = 128
USE_LAYER_7 = True

LOG_NAME = f"jvs_adan_aam_h{int(HIDDEN_SIZE)}_b{int(BATCH_SIZE)}_e{int(NUM_EPOCHS)}_s{int(SCHEDULER_T_INITIAL)}_kl"
if not USE_LAYER_7:
    LOG_NAME = LOG_NAME + "_no_layer7"

logger.info(f"LOG_NAME: {LOG_NAME}")

# ----------------------------
# seed
SEED = 3407
seed_everything(SEED, workers=True)

config_path = Path(__file__).resolve().parent.parent.parent / "src/conf"
@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def train(cfg: DictConfig):
    cfg.dataset.train.dataset_list = TRAIN_DATASET_LIST
    
    cfg.ml.num_epochs = NUM_EPOCHS
    cfg.ml.batch_size = BATCH_SIZE
    cfg.ml.scheduler.t_initial = SCHEDULER_T_INITIAL
    cfg.model.ecapa_tdnn.hidden_size = HIDDEN_SIZE
    cfg.model.ecapa_tdnn.use_layer7 = USE_LAYER_7
    cfg.dataset.augment.time_stretch_params = AUGMENT_TIME_STRETCH_PARAMS

    logger.info(cfg)
    
    ################################
    # データセットとモデルの設定
    ################################
    _train_audio_list = collect_dataset_from_dataset_list(
                cfg.dataset.train.dataset_list, cfg.dataset.train.spk_index_info_json_path
            )
    _test_audio_list = collect_dataset_from_dataset_list(
        cfg.dataset.test.dataset_list, cfg.dataset.test.spk_index_info_json_path
    )
    dataset = VoiceDataModule(cfg, _train_audio_list, _test_audio_list)
    model = EcapaTdnnKLModelModule(cfg, _train_audio_list)
    
    ################################
    # コールバックなど訓練に必要な設定
    ################################
    # ロガー
    tflogger = TensorBoardLogger(save_dir=LOG_SAVE_DIR, name=LOG_NAME, version=EXP_ID)
    tflogger.log_hyperparams(cfg)
    # モデル保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_SAVE_DIR,
        filename="checkpoint-{epoch:04d}-{val_eer:.4f}",
        save_top_k=cfg.ml.model_save.top_k,
        monitor=cfg.ml.model_save.monitor,
        mode=cfg.ml.model_save.mode
    )
    
    
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    if cfg.ml.early_stopping.use:
        # 早期終了の追加
        from pytorch_lightning.callbacks import EarlyStopping
        callback_list.append(EarlyStopping(
            monitor=cfg.ml.early_stopping.monitor,
            mode=cfg.ml.early_stopping.mode,
            patience=cfg.ml.early_stopping.patience
        ))
        
        
     ################################
    # 訓練
    ################################
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
            precision=cfg.ml.mix_precision,
            gradient_clip_val=cfg.ml.gradient_clip_val,
            accelerator=device,
            devices=cfg.ml.gpu_devices,
            max_epochs=cfg.ml.num_epochs,
            accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
            profiler=cfg.ml.profiler,
            fast_dev_run=FAST_DEV_RUN,
            logger=tflogger,
            callbacks=callback_list
        )
    logger.debug("START TRAIN")
    trainer.fit(model, dataset)
    
    # model チェックポイントの保存
    best_model_path = f"{MODEL_SAVE_DIR}/best_model.ckpt"
    if len(checkpoint_callback.best_model_path):
        logger.info(f"BEST MODEL: {checkpoint_callback.best_model_path}")
        logger.info(f"BEST SCORE: {checkpoint_callback.best_model_score}")
        _ckpt = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(_ckpt["state_dict"])
        torch.save(model.model.state_dict(), best_model_path)
        logger.info(f"To BEST MODEL: {best_model_path}")
        # FOR LOAD
        # _ckpt = torch.load(f"{cfg.ml.model_save.save_dir}/{cfg.ml.log_name}/{cfg.ml.version}/best_model.ckpt")
        # model.model.load_state_dict(_ckpt)
    else:
        print("best model is not exist.")
        
if __name__ == "__main__":
    train()