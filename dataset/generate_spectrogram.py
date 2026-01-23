import os, yaml, torchaudio, numpy as np
from nnAudio import Spectrogram
import torch.nn.functional as F
from tqdm import tqdm

# Pad MFCC to get same shape (95,126)
def pad_mfcc(mfcc_out):
    return F.pad(mfcc_out, (0, 0, 0, 82))

# Function to create spectrograms
def create_spectrograms(file_path, config):
    sample_rate = config["dataset"]["sample_rate"]
    waveform, sample_rate = torchaudio.load(file_path)

    #CQT
    cqt_spectrogram = Spectrogram.CQT(
        sr=sample_rate, hop_length=config["dataset"]["HOP_LENGTH"], fmin=config["dataset"]["FMIN"], fmax=config["dataset"]["FMAX"],
        n_bins=config["dataset"]["FREQ_BINS"], bins_per_octave=12, filter_scale=1, norm=1,
        window='hann', center=True, pad_mode='reflect', trainable=False,
        output_format='Magnitude', verbose=False
    )(waveform)

    #Gammatone
    gamma_spectrogram = Spectrogram.Gammatonegram(
        sr=sample_rate, n_fft=config["dataset"]["N_FFT"], n_bins=config["dataset"]["FREQ_BINS"], hop_length=config["dataset"]["HOP_LENGTH"],
        window='hann', center=True, pad_mode='reflect',
        power=2.0, htk=False, fmin=config["dataset"]["FMIN"], fmax=config["dataset"]["FMAX"], norm=1,
        trainable_bins=False, trainable_STFT=False, verbose=False
    )(waveform)

    # MFCC
    mfcc_transform = Spectrogram.MFCC(
        sr=config["dataset"]["sample_rate"], n_mfcc=config["dataset"]["N_MFCC"], n_fft=config["dataset"]["N_FFT"],
        win_length=config["dataset"]["N_FFT"],  hop_length=config["dataset"]["HOP_LENGTH"], window='hann',
        fmin=config["dataset"]["FMIN"],fmax=config["dataset"]["FMAX"],power=2.0,verbose=False
    )(waveform)
    mfcc_transform = pad_mfcc(mfcc_transform)

    return cqt_spectrogram, gamma_spectrogram, mfcc_transform

# Function to process audio files in the directory structure
def process_audio_files(config):
    base_dir = config["dataset"]["root_dir"]
    for inclusion_folder in os.listdir(base_dir):
        print(f'\n\nProcessing scenario:{inclusion_folder}\n\n')
        inclusion_path = os.path.join(base_dir, inclusion_folder)
        if not os.path.isdir(inclusion_path):
            continue

        for split in ["test", "validation", "train"]:
            print(f'\n\nProcessing split:{inclusion_folder}\n\n')
            split_path = os.path.join(inclusion_path, split)
            audio_path = os.path.join(split_path, "audio")

            # Prepare output directories
            cqt_path = os.path.join(split_path, "cqt")
            gamma_path = os.path.join(split_path, "gamma")
            mfcc_path = os.path.join(split_path, "mfcc")

            for output_dir in [cqt_path, gamma_path, mfcc_path]:
                os.makedirs(output_dir, exist_ok=True)

            for subfolder in os.listdir(audio_path):
                subfolder_path = os.path.join(audio_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    print(f'\n\nnot found: {subfolder_path}\n\n')
                    continue

                # Create corresponding subfolders in output directories
                cqt_subfolder = os.path.join(cqt_path, subfolder)
                gamma_subfolder = os.path.join(gamma_path, subfolder)
                mfcc_subfolder = os.path.join(mfcc_path, subfolder)

                for output_subfolder in [cqt_subfolder, gamma_subfolder, mfcc_subfolder]:
                    os.makedirs(output_subfolder, exist_ok=True)

                # Process audio files
                for file in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder} in {split_path}"):
                    if file.endswith(".wav"):
                        file_path = os.path.join(subfolder_path, file)

                        # Generate spectrograms
                        cqt, gamma, mfcc = create_spectrograms(file_path,config)

                        # Save spectrograms as .npy files
                        np.save(os.path.join(cqt_subfolder, file.replace(".wav", ".npy")), cqt)
                        np.save(os.path.join(gamma_subfolder, file.replace(".wav", ".npy")), gamma)
                        np.save(os.path.join(mfcc_subfolder, file.replace(".wav", ".npy")), mfcc)


if __name__ == "__main__":
    # Load configuration    
    with open("../config/mobilenet.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    # Process all audio files and generate spectrograms folders for each audio file
    process_audio_files(config)