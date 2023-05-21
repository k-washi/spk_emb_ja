import json

from src.dataset.utils import get_audio_file_num_by_spk_index_info
from src.criteria.aamsoftmax import AAMSoftmax

def get_loss(
    loss_type,  
    spk_index_info_json_path, 
    use_ce_weight:bool=False, 
    audio_file_list=[],
    hidden_size:int=64,
    aam_margin:float=0.2,
    aam_scale:float=30,
):
    with open(spk_index_info_json_path, 'r') as f:
        spk_index_info = json.load(f)
    
    # ce weight create
    ce_weight = None
    if use_ce_weight:
        audio_file_num_by_spk_index = get_audio_file_num_by_spk_index_info(audio_file_list)
        assert len(audio_file_num_by_spk_index) == len(spk_index_info), \
            f"audio_file_num_by_spk_index: {len(audio_file_num_by_spk_index)} is not equal to number of spk_index_info: {len(spk_index_info)}"
        ce_weight = [1] * len(audio_file_num_by_spk_index)
        for i, data_num in enumerate(audio_file_num_by_spk_index):
            ce_weight[i] = ce_weight[i] / data_num
    
    if loss_type == 'aam':
        return AAMSoftmax(len(spk_index_info), hidden_size=hidden_size, m=aam_margin, s=aam_scale, cross_entropy_weight=ce_weight)
    else:
        raise NotImplementedError(f"loss_type: {loss_type} is not implemented")
            