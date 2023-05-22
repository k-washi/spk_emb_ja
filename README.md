# 環境構築

```
docker-compose up -d
```

## Docker内

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

実験実行例

```
python ./pipeline/train/exp002.py
```

# データ形式

```
    音声データセットdir
     |-spk1
     |  |-wav
           |-audio_file_00001.wav
           |-audio_file_00002.wav
```


# tesnsorboard

```
tensorboard --logdir=./logs/ --host=0.0.0.0 --port=18382
```

# Refarence

- [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

