import os
import glob
import argparse
from audio_phrase_extraction.extract import RemoveNoise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('path_in', help='the path to input directory with audio')
    parser.add_argument('path_out', help='the path of the noise')
    parser.add_argument('--db', type=int, choices=range(-100, 2), default=-90, help='db (from -100 to 1)')
    parser.add_argument('--min_len_noise', type=int, default=10, help='min length a noise in mc')
    parser.add_argument('--seek_step', type=int, default=1, help='seek step in ms')


    # Parse arguments, first mandatory then not.
    args = parser.parse_args()

    # Take the wav-files paths
    file_paths = glob.glob(os.path.join(args.path_in, '*.wav'))
    files = [os.path.split(path)[-1] for path in file_paths]

    # Check `path_out` exists
    try:
        os.mkdir(args.path_out)
    except FileExistsError:
        pass

    # Create arg-dict
    if file_paths:
        argsnoise = {'path_in': args.path_in,
                    'path_out': args.path_out,
                    'min_silence_len_noise': args.min_len_noise,
                    'db': args.db,
                    'seek_step': args.seek_step,
                    'files_in': file_paths,
                    'files': files}



    remNoise = RemoveNoise(**argsnoise)
    files_noNoise = remNoise.removeNoise()




