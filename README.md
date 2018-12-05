# OpenBCI data processing for subvocal recognition
- This is the repo to turn OpenBCI raw data into spectrograms and to train the model on said data.

#### `1_openbci_to_wavfile.ipynb`
- Checks the latest recordings in the SavedData folder from OpenBCI, takes the array of values of each channel
- and turns it into a wavfile.  

#### `2_wav_to_spectrogram.ipynb`
- Before executing this code you first need to split the data using the `process_audio.py` from `subvocalization_stream`,
- once you split the wavfiles, this code transforms the individual wavfiles into sepctrograms.

#### `3_spectrogram_to_csv.ipynb`
- This script will get the file location of each spectrogram and list them into a csv file for training the model.

#### `4_subvocal_train_model.ipynb`
- This script will train the model using the assigned csv file.

#### `5_subvocal_model_find_outliers.ipynb`
- Using the trained model we can apply it to the dataset to find outliers and create a csv file to exclude said outlier data.
