import os
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_silence


class RemoveNoise():
    def __init__(self, **args):
        self.path_in = args['path_in']
        self.path_out = args['path_out']
        self.files = args['files']
        self.msilnoise = args['min_silence_len_noise']
        self.db = args['db']
        self.seek_step = args['seek_step']
        self.files_in = args['files_in']

    def _joinPath(self, path, dir):
        """
        :param path: dir path
        :param dir: subdir
        :return: path + subdir

        """
        return os.path.join(path, dir)

    def _noNoisedir(self):
        if not os.path.isdir(self._joinPath(self.path_out, 'spec_noise')):
            noise_out = self._joinPath(self.path_out, 'spec_noise')
            os.mkdir(noise_out)
        else:
            noise_out = self._joinPath(self.path_out, 'spec_noise')
            print(f"Dir {noise_out} exists")
        noise_out = os.path.join(noise_out, 'spec_noise.wav')
        return noise_out

    def _prepNoiseDir(self):
        """
        Prepare the directory and path for noise
        :return: files_noNoise - lst of files, noise_out - path to noise

        """
        path_noNoise = self._joinPath(self.path_in, 'Audio_without_noise')
        os.mkdir(path_noNoise) if not os.path.isdir(path_noNoise) else print(f"Dir \
        {path_noNoise} exists")

        files_noNoise = [os.path.join(path_noNoise, file) for file in self.files]
        noise_out = self._noNoisedir()
        return files_noNoise, noise_out

    def _getNoise(self, sound, db):
        """
        :param sound: audio-sound
        :return: slices of noise

        """
        noise_int = detect_silence(sound,
                               min_silence_len=self.msilnoise,
                               silence_thresh=db,
                               seek_step=self.seek_step)
        return noise_int

    def _concatenate_audio(self, pieces):
        """
        Return the concatenated pieces of sounds.

        """
        out = 0
        for piece in pieces:
            out += piece
        return out

    def _getNoiseEx(self, sound, intervals):
        """
        Extract noiese piaces from the orinal sound.
        Return the concatenated pieces of noise sound.

        """
        return self._concatenate_audio([sound[i:j] for i, j in intervals])

    def _saveNoise(self, fpathIn):
        """
        Takes the audio => extract niose => save noise into special folder.
        :param fpathIn: path to the audio
        :return: None

        """
        counter = 0  # Prevent infinity loop
        sound = AudioSegment.from_file(fpathIn,
                                       format="wav")

        # Get noise from audio.
        noise_int = self._getNoise(sound, self.db)
        while counter < 6 and not noise_int:
            counter += 1
            if counter == 1:
                new_db = self.db + 10
            else:
                new_db = new_db + 10
                noise_int = self._getNoise(sound, new_db)

        sound_noise = self._getNoiseEx(sound, noise_int)
        try:
            sound_noise.export(self.noise_out, format="wav")
        except:
            print(f"Noise for {fpathIn} wasn't found")

    def removeNoise(self):
        """
        Takes noise from spec. folder and delete it from relative
        :return: None

        """
        self.files_noNoise,self.noise_out = self._prepNoiseDir()
        for file_path_in, file_path_out in zip(self.files_in, self.files_noNoise):
            # Save noise from audio.
            self._saveNoise(file_path_in)

            # Clean audio from noise with respect to n. channels.
            data, rate = sf.read(file_path_in)
            noise, rate_noise = sf.read(self.noise_out)
            if len(data.shape) == 1:
                n_channels = 1
                new_data = nr.reduce_noise(audio_clip=data, noise_clip=noise,
                                           prop_decrease=1.0)
            else:
                n_channels = data.shape[1] # Get number of channels
                new_data = np.array([nr.reduce_noise(audio_clip=data[:, idx_ch],
                                                     noise_clip=noise[:,idx_ch],
                                                     prop_decrease=1.0) \
                                     for idx_ch in range(n_channels)]).T
            sf.write(file_path_out, new_data, rate)
        print("Noise has been deleted")
        return self.files_noNoise

