import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Union, List, Dict

@dataclass
class SpkAttribute:
    data_num: int

def collect_dataset_from_dataset_list(dataset_dir_list:List[Union[str, Path]], spk_index_info_json_output_path: str) -> List[Tuple[int, str]]:
    """データセットのディレクトリ群から、audioファイルを集めて、(ラベル, ファイルパス)のリストを作成
    
    Args:
        dataset_dir_list (List[Union[str, Path]]): データセットのディレクトリのリスト
        spk_index_info_json_output_path (str): {spk_name: label_index}のjsonを作成市保存するパス

    Returns:
        List[Tuple[int, str]]: (ラベル, ファイルパス)のリスト
    """
    spk_attr_dic, spk_index_info, tmp_dataset_list, dataset_list = {}, {}, [], []
    for dataset_dir in dataset_dir_list:
        _spk_attr_dic, _dataset_list = collect_audio_files_from_dataset(dataset_dir)
        spk_attr_dic.update(_spk_attr_dic)
        tmp_dataset_list.extend(_dataset_list)
    
    # spkにindexを割り振る
    spk_list = sorted(list(spk_attr_dic.keys()))
    for i, spk_name in enumerate(spk_list):
        spk_index_info[spk_name] = i
    spk_index_info_json_path = Path(spk_index_info_json_output_path)
    spk_index_info_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spk_index_info_json_path, 'w') as f:
        json.dump(spk_index_info, f, indent=4)
    
    # ラベルをindexに変更
    for spk_name, audio_file_path in tmp_dataset_list:
        dataset_list.append((int(spk_index_info[spk_name]), str(audio_file_path)))
    
    return dataset_list

def get_audio_file_num_by_spk_index(audio_file_list:List) -> Dict[int, int]:
    """spkごとのデータ数を計算
    collect_dataset_from_dataset_listの結果を入力
    Args:
        audio_file_list (List[Tuple): (ラベル, ファイルパス)のリスト

    Returns:
        List[int]: 各label indexごとのデータ数のリスト
    """
    audio_file_num_by_spk = {}
    for label, _ in audio_file_list:
        if label not in audio_file_num_by_spk:
            audio_file_num_by_spk[label] = 1
        else:
            audio_file_num_by_spk[label] += 1
    audio_file_num_by_spk_list = [0] * len(audio_file_num_by_spk)
    try:
        for k, v in audio_file_num_by_spk.items():
            audio_file_num_by_spk_list[k] = v
    except Exception as e:
        print(k)
        print(v)
        raise ValueError(e)
    
    return audio_file_num_by_spk_list
    
    
    
def collect_audio_files_from_dataset(dataset_dir_path:Union[str, Path]) -> Tuple[Dict[str, SpkAttribute], List[Tuple[str, str]]]:
    """データセットディレクトリから音声ファイル情報を取得
    データセットの形式
    
    音声データセットdir (dataset_dir_path)
     |-spk1
     |  |-wav
           |-audio_file_00001.wav
           |-audio_file_00002.wav

    Args:
        dataset_dir_path (Union[str, Path]): データセットのディレクトリ

    Returns:
        spk attrDict[str, SpkAttribute]: {"spk name": SpkAttribute}の話者情報
        
        List[Tuple[str, str]]: 音声ファイルのリスト [(spk name, audio file path)]
    """
    dataset_dir_path = Path(dataset_dir_path)
    spk_dir_list = list(dataset_dir_path.glob("*"))
    attr_dic = {}
    dataset_list = []
    for spk_dir in spk_dir_list:
        spk_name = spk_dir.stem
        spk_dir = spk_dir / "wav"
        audio_file_list = sorted(list(spk_dir.glob("*.wav")) + list(spk_dir.glob("*.mp3")))
        assert len(audio_file_list) > 0, f"{spk_name}の音声ファイルが見つかりませんでした。"
        attr_dic[spk_name] = SpkAttribute(len(audio_file_list))
        for audio_file_path in audio_file_list:
            dataset_list.append((spk_name, str(audio_file_path)))
    
    return attr_dic, dataset_list

if __name__ == "__main__":
    # collect_audio_files_from_dataset
    attr_dic, dataset_list = collect_audio_files_from_dataset("/data/jvs_vc")
    print(attr_dic["jvs001"], len(attr_dic))
    print(dataset_list[:5])
    # > SpkAttribute(data_num=130) 100
    # > [('jvs063', '/data/jvs_vc/jvs063/wav/BASIC5000_0190.wav'), ('jvs063', '/data/jvs_vc/jvs063/wav/BASIC5000_0239.wav'), ('jvs063', '/data/jvs_vc/jvs063/wav/BASIC5000_0357.wav'), ('jvs063', '/data/jvs_vc/jvs063/wav/BASIC5000_0474.wav'), ('jvs063', '/data/jvs_vc/jvs063/wav/BASIC5000_0486.wav')]
    
    
    # collect_dataset_from_dataset_list
    dataset_list = ["/data/jvs_vc", "/data/common_voice"]
    audio_file_list = collect_dataset_from_dataset_list(dataset_list, "results/sample_spk_index_info.json")
    
    audio_file_num_by_spk_index = get_audio_file_num_by_spk_index(audio_file_list)
    print(len(audio_file_num_by_spk_index), min(audio_file_num_by_spk_index), max(audio_file_num_by_spk_index))