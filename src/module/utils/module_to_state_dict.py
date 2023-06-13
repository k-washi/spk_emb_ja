import torch
from pathlib import Path
from src.ecapa_tdnn.model import ECAPA_TDNN
from src.module.model.base import EcapaTdnnModelModule
from src.utils.conf import get_hydra_cnf

def convert_checkpoint_to_st(input_path, output_path, hidden_size):
    cfg = get_hydra_cnf("src/conf", "config")
    cfg.model.ecapa_tdnn.hidden_size = hidden_size
    model = EcapaTdnnModelModule.load_from_checkpoint(input_path, cfg=cfg, strict=False)
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.model.state_dict(), output_path)
    
    model = ECAPA_TDNN(
        cfg.model.ecapa_tdnn.channel_size,
        cfg.model.ecapa_tdnn.hidden_size
    )
    model.load_state_dict(torch.load(output_path))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        "学習した重みから、state_dictのみ抽出する"
    )
    
    
    parser.add_argument("--input_checkpoint", default="checkpoints/00010/checkpoint-epoch=0120-train_acc=73.1927-val.ckpt")
    parser.add_argument("--output_checkpoint", default="checkpoints/st/ecapa_tdnn_ja_best.pth")
    parser.add_argument("--hidden_size", default=128, type=int)
    args = parser.parse_args()
    convert_checkpoint_to_st(args.input_checkpoint, args.output_checkpoint, args.hidden_size)