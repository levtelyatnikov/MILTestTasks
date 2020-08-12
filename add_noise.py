import os
import argparse
import numpy as np
import soundfile as sf
from noise.noise_functional import add_noise

from pathlib import Path

np.random.seed(0)


if __name__ == "__main__":
    # Prepare the dataset.
    parser = argparse.ArgumentParser()
    parser.add_argument('path_in', help='the path to input directory with audio')
    parser.add_argument('sub_size', type=float, help='precent from the initial size')

    args = parser.parse_args()
    size_percent = args.sub_size
    paths = list(map(str,(Path(args.path_in).rglob('*.flac'))))
    out = np.random.choice(paths, size=int(len(paths)*size_percent), replace=False)

    with open('audio_paths.txt','w') as f:
        for path in out:
            f.write(path+"\n")

    print(f"Audio subsample = {len(out)} out from {len(paths)}")


    with open('audio_paths.txt','r') as f:
        data_paths = f.read().splitlines()

    try:
        os.mkdir('NoisyData')
    except FileExistsError:
        pass


    for path in data_paths:
        data,samplerate = add_noise(path)
        pathout = "NoisyData/"+os.path.split(path)[-1].strip('.flac') + '.wav'
        sf.write(pathout, data, samplerate)

