import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.utils.logger import get_logger
logger = get_logger(debug=True)

config_path = Path(__file__).resolve().parent.parent / "src/conf"
@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def train(cfg: DictConfig):
    print(cfg)

if __name__ == "__main__":
    train()