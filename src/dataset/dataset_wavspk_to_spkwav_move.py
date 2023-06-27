from pathlib import Path
import shutil
from tqdm import tqdm
def move_wav_file(input_dir, output_dir):
    input_dir = Path(input_dir)
    spk_dirs = sorted(list(input_dir.glob("*")))
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for spk_dir in tqdm(spk_dirs):
        output_spk_dir = output_dir / f"jtube_{spk_dir.name}"
        output_spk_dir.mkdir(exist_ok=True, parents=True)
        output_spk_dir = output_spk_dir / "wav"
        output_spk_dir.mkdir(exist_ok=True, parents=True)
        
        audio_files = sorted(list(spk_dir.glob("*")))
        for audio_file in audio_files:
            output_path = output_spk_dir / audio_file.name
            shutil.copy(str(audio_file), str(output_path))
        
        
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="move spk id in wav dir to wav in spk id dir")
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)

    args = parser.parse_args()
    
    move_wav_file(args.input_dir, args.output_dir)