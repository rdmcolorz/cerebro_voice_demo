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
from predictModel import *
import pygame
from ouimeaux.environment import Environment
tf.logging.set_verbosity(tf.logging.ERROR)
# make wav file VARS ######################
FORMAT = pyaudio.paInt16
CHANNELS = 8
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 1
DEVICE_ID = 2 # UMC 1820 index is 2 on my mac, use audio.get_device_info_by_index(i)
WAVE_OUTPUT_FILENAME = "./audio/predict/file.wav"

# make images VARS  ####################
ROOT = os.getcwd() + "/audio/"
IMG_ROOT = os.getcwd() + "/images_scaled/"
pattern = "[0-9]{2}_[0-9]{2}"
VALID_LABELS = ["yes", "no", "stop", "go", "right", "left", "down", "up", "on", "off", "test"]
IMG_EXT = ".png"
VERBOSITY = 1000
# make images VARS  ####################
# PATHS
paths = {
    "Training":ROOT+"paths_scaled_combined.csv",
    "Model": ROOT+"demoModelOutliers",
    "Logs":RUN_ROOT_LOG+"{}_{}/".format(NUMS, datetime.strftime(curr_time(), "%b%d%Y_%H%M%S"))
}
paths["Log"] = paths["Logs"] + "log.txt"
if not os.path.isdir(RUN_ROOT):
    os.mkdir(RUN_ROOT)
if not os.path.isdir(RUN_ROOT_LOG):
    os.mkdir(RUN_ROOT_LOG)
if not os.path.isdir(paths["Logs"]):
    os.mkdir(paths["Logs"])

#######################################################
OUTPUT = os.getcwd() + "/demo.csv"
NUM_CHANNELS = 22
CATS, MONTHS, DAYS, LABELS, SEQ, SETS = [], [], [], [], [], []

########################################
CATEGORY = ["no_voice"]
LABELS = ["yes", "no"]
CHANNELSX = [1, 2, 3, 5, 6, 7, 8]
NUMS = ''.join([str(x) for x in CHANNELSX])
MONTHS = [9]
DAYS = [25]

###########################################
#			MAKE WAV FILES   		      #
###########################################


while True:
    audio = pyaudio.PyAudio()
    # start Recording
    stream = audio.open(format=FORMAT, 
                        channels=CHANNELS,
                        rate=RATE, 
                        input=True,
                        input_device_index=DEVICE_ID,
                        frames_per_buffer=CHUNK)
    print("-" * 21 + "\nRecording...\n" + "-" * 21)
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        rawsamps = stream.read(CHUNK)
        frames.append(rawsamps)

    print("-" * 21 + "\nDone Recording!\n" + "-" * 21)

    # stop Recording
    stream.stop_stream()
    stream.close()
    # audio.terminate()

    # write("eight.wav", 44100, data)
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    b1 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch1/0000.wav remix 1"
    b2 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch2/0000.wav remix 2"
    b3 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch3/0000.wav remix 3"
    b4 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch4/0000.wav remix 4"
    b5 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch5/0000.wav remix 5"
    b6 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch6/0000.wav remix 6"
    b7 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch7/0000.wav remix 7"
    b8 = "sox ./audio/predict/file.wav ./audio/no_voice/09_25/yes/ch8/0000.wav remix 8"
    commands = [b1,b2,b3,b4,b5,b6,b7,b8]
    for i in range(8):
        subprocess.run(commands[i], shell=True)

    ###########################################
    #			MAKE SPECTROGRAMS and csv     #
    ###########################################
    if not os.path.isdir(IMG_ROOT):
        os.mkdir(IMG_ROOT)

    dir_pairs = {
        ROOT+"voice":IMG_ROOT+"voice",
        ROOT+"no_voice":IMG_ROOT+"no_voice"
    }

    for input_dir in dir_pairs:
        output_dir = dir_pairs[input_dir]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        timer(process, input_dir, output_dir)

    df = timer(make_df_from_images, IMG_ROOT)
    df.to_csv(OUTPUT, index=False)

    ###########################################
    #			PREDICT                       #
    ###########################################

    count = 0
    def _parse_function(label, *filenames):
        global count
        count += 1
        if count % VERBOSITY == 0:
            print_and_log("\tProcessed {}th image".format(count))
        expected_shape = tf.constant([1, INPUT_HEIGHT, INPUT_WIDTH, IMG_CHANNELS])
        image = None
        for filename in filenames:
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_image(image_string, channels=IMG_CHANNELS)
            image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
            image_decoded = tf.reshape(image_decoded, expected_shape)
            image_decoded = tf.image.rgb_to_grayscale(image_decoded)
            if RESIZE:
                image_decoded = tf.image.resize_bicubic(image_decoded, [TARGET_HEIGHT, TARGET_WIDTH])
            if image is not None:
                image = tf.concat([image, image_decoded], 3)
            else:
                image = image_decoded
        return image, label

    with open(paths["Log"], 'w') as log:
        log.write(make_header("Starting Script\n"))

    # Create variables for the paths
    train_csv = paths["Training"]

    # Store the labels to train
    all_labels = LABELS
    labels = ["yes", "no", "stop", "unknown"]
    num_labels = len(labels) - 1
    labels = {x[1]:x[0] for x in enumerate(labels)}
    reverse_lookup = {labels[k]:k for k in labels}

    # print_and_log_header("MAKING TRAINING DATA")
    demo_data = pd.read_csv("./demo.csv")

    demo_data = demo_data.drop(columns=['Path4'])

    # Filter the training data
    demo_data = select_categories(demo_data, CATEGORY)
    # demo_data = select_channels(demo_data, CHANNELS)
    demo_data = select_labels(demo_data, LABELS)
    demo_data = select_months(demo_data, MONTHS)
    demo_data = select_days(demo_data, DAYS)
    # train_data = remove_voice(train_data)
    demo_data = demo_data.sample(frac=1).reset_index(drop=True)
    tdcopy = pd.DataFrame(demo_data)
    demo_data["Label"] = demo_data["Label"].map(labels)

    # if VERBOSE:
    #     print_and_log_header("TRAIN DATA")
    #     print_and_log(demo_data.describe())
    #     print_and_log(demo_data.head(10))

    # # Separate Labels
    demo_labels = demo_data.pop(target_label)
    img_paths = ["Path{}".format(channel) for channel in CHANNELSX]
    demo_data = demo_data[img_paths]

    # Vectors of filenames.
    t_f, v_f, s_f = [], [], []
    for i in range(1, 1 + len(CHANNELSX)):
        channel = CHANNELSX[i-1]
        l = "Path{}".format(channel)
        t_f.append(tf.constant(demo_data[l]))

    # `labels[i]` is the label for the image in `filenames[i]
    # Vectors of labels
    demo_labels = tf.constant(demo_labels)

    # Make datasets from filenames and labels
    demo_data = tf.data.Dataset.from_tensor_slices((demo_labels, *t_f))
    demo_data = timer(lambda: demo_data.map(_parse_function))
    # Create the Estimator
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=paths["Model"])
    # Create the input functions.
    demo_eval_input_fn = create_predict_input_fn(demo_data, DEFAULT_BS)

    results = [x for x in classifier.predict(input_fn=demo_eval_input_fn)]
    classes = [x["classes"] for x in results]
    probs = [x["probabilities"] for x in results]

    result = classes[0]

    # Turn wemo device on and off
    on = "wemo switch \"cerebro plug\" on"
    off = "wemo switch \"cerebro plug\" off"
    on_off = [on, off]

    if result < 2:
        subprocess.run(on_off[result], shell=True)
        if result == 0:
            print("-" * 21 + "\nLights ON!\n" + "-" * 21)
        else:
            print("-" * 21 + "\nLights OFF!\n" + "-" * 21)
    else:
        print("-" * 21 + "\nNothing Happened :(\n" + "-" * 21)

# pygame.init()

# pygame.display.set_caption('CerebroVoice')
# size = [1280, 960]
# screen = pygame.display.set_mode(size)
 
# clock = pygame.time.Clock()
 
# basicfont = pygame.font.SysFont(None, 300)
# print(len(classes))
# predict_labels = ['yes', 'no', 'unknown']
# print_this = predict_labels[classes[0]]
# text = basicfont.render(print_this, True, (255, 0, 0), (255, 255, 255))
# textrect = text.get_rect()
# textrect.centerx = screen.get_rect().centerx
# textrect.centery = screen.get_rect().centery
 
# screen.fill((255, 255, 255))
# screen.blit(text, textrect)
 
# pygame.display.update()
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
#     pygame.display.flip()