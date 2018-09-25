import pyaudio
import wave
import subprocess
from scipy.io.wavfile import write
import os, shutil, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
from helpers import *
from spectrogram_func import *

ROOT = os.getcwd() + "/audio/"
IMG_ROOT = os.getcwd() + "/images_scaled/"
pattern = "[0-9]{2}_[0-9]{2}"
VALID_LABELS = ["yes", "no", "stop", "go", "right", "left", "down", "up", "on", "off", "test"]
IMG_EXT = ".png"
VERBOSITY = 1000
OUTPUT = os.getcwd() + "/test.csv"
NUM_CHANNELS = 22
CATS, MONTHS, DAYS, LABELS, SEQ, SETS = [], [], [], [], [], []
for i in range(1, NUM_CHANNELS+1):
    globals()["PATH{}".format(i)] = []

def preprocess(samples, sample_rate, multiplier=1):
    sr = sample_rate * multiplier
    padded = np.zeros(sr)
    samples = samples[:sr]
    padded[:samples.shape[0]] = samples
    return padded

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def process(input_dir, output_dir, overwrite=False):
    items = 0
    created = 0
    found = 0
    date_mult = {"08_08":2, "08_11":2, "09_20":2}
    plt.ioff()
    for date in [x for x in os.listdir(input_dir) if re.match(pattern, x)]:
        multiplier = 1
        if date in date_mult:
            multiplier = date_mult[date]
        date_path = os.path.join(input_dir, date)
        o_date_path = os.path.join(output_dir, date)
        make_dir(o_date_path)
        for label in [d for d in os.listdir(date_path) if d in VALID_LABELS]:
            label_path = os.path.join(date_path, label)
            o_label_path = os.path.join(o_date_path, label)
            make_dir(o_label_path)
            print("\tProcessing: {}".format(label_path))
            print("\tTime: {}".format(curr_time()))
            for channel in [d for d in os.listdir(label_path) if d.startswith("ch")]:
                voice = False
                ch = int(channel[2:])
                if ch == 4 or ch >= 9:
                    voice = True
                channel_path = os.path.join(label_path, channel)
                o_channel_path = os.path.join(o_label_path, channel)
                make_dir(o_channel_path)
                channel_num = channel[-1]
                for file in [f for f in os.listdir(channel_path) if f.endswith(".wav")]:
                    items += 1
                    wavpath = os.path.join(channel_path, file)
                    imgpath = os.path.join(o_channel_path, file[:-4] + IMG_EXT)
                    if overwrite or not os.path.isfile(imgpath):
                        created += 1
                        if items % VERBOSITY == 0:
                            print("\t\tCreated {}th image".format(items))
                        sample_rate, samples = wavfile.read(wavpath)
                        samples = preprocess(samples, sample_rate, multiplier)
#                         freqs, times, spectrogram = signal.spectrogram(samples, sample_rate)
                        if voice:
                            S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
                        else:
                            S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128, fmax=512)
                        log_S = librosa.power_to_db(S, ref=np.max)
                        fig = plt.figure(figsize=(1.28, 1.28), dpi=100, frameon=False)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        plt.axis('off')
                        librosa.display.specshow(log_S)                          
                        plt.savefig(imgpath)
                        plt.close()
                    else:
                        found += 1
                        if items % VERBOSITY == 0:
                            print("\t\tFound {}th image".format(items))
    print("\tFound:\t\t{}\n\tCreated:\t{}".format(found, created))
    plt.ion()

def make_df_from_images(image_root):
    for cat in [d for d in os.listdir(image_root) if "voice" in d]:
        cat_path = os.path.join(image_root, cat)
        for date in [d for d in os.listdir(cat_path) if re.match(pattern, d)]:
            print("\tProcessing {}".format(date))
            date_path = os.path.join(cat_path, date)
            month = int(date[:2])
            day = int(date[3:])
            date_count = 0
            for label in [d for d in os.listdir(date_path) if d in VALID_LABELS]:
                label_path = os.path.join(date_path, label)
                placeholder = os.path.join(label_path, "ch1")
                for image in [f for f in os.listdir(placeholder) if f.endswith(IMG_EXT)]:
                    date_count += 1
                    for i in range(1, NUM_CHANNELS+1):
                        p = os.path.join(label_path, "ch{}".format(i), image)
                        if os.path.exists(p):
                            globals()["PATH{}".format(i)].append(p)
                        else:
                            globals()["PATH{}".format(i)].append(float('nan'))
                    CATS.append(cat)
                    DAYS.append(day)
                    MONTHS.append(month)
                    LABELS.append(label)
                    sequence_number = int(image[:-4])
                    basenum = sequence_number % 10
                    SEQ.append(sequence_number)
                    if basenum < 8:
                        SETS.append("Training")
                    elif basenum < 9:
                        SETS.append("Validation")
                    else:
                        SETS.append("Testing")
            print("\t\tProcessed {} sequences".format(date_count))
    d = {
            "Category":CATS,
            "Day":DAYS,
            "Month":MONTHS,
            "Label":LABELS,
            "SequenceNumber":SEQ,
            "Set":SETS
        }
    for i in range(1, NUM_CHANNELS+1):
        d["Path{}".format(i)] = globals()["PATH{}".format(i)]
    return pd.DataFrame(d)