
import os
import glob
import fnmatch
import pandas as pd
import numpy as np
import librosa  # To deal with Audio files
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import IPython.display as ipd
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
import joblib
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Concatenate, Conv1D, Conv2D, SeparableConv1D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Input, add, Flatten, Dense, BatchNormalization, Dropout, LSTM, GRU
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, Activation, LeakyReLU, ReLU
from tensorflow.keras import regularizers, backend as K
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



data_path = "/aiheartbeat"
print(os.listdir(data_path))

train_data      = data_path 
unlabel_data    = data_path  + "/Unlabelled/"

normal_data     = train_data + '/Normal/'
murmur_data     = train_data + '/Murmur/'
extrastole_data = train_data + '/Extrastole/'
artifact_data   = train_data + '/Artifact/'
extrahls_data   = train_data + "/Extrahls/"

print("Normal files:", len(os.listdir(normal_data))) #length of normal training sounds
print("Murmur files:",len(os.listdir(murmur_data))) #length of murmur training sounds 
print("Extrastole files", len(os.listdir(extrastole_data))) #length of extrastole training sounds 
print("Artifact files:",len(os.listdir(artifact_data))) #length of artifact training sounds 
print("Extrahls files:",len(os.listdir(extrahls_data))) #length of extrahls training sounds 

print('TOTAL TRAIN SOUNDS:', len(os.listdir(normal_data)) 
                              + len(os.listdir(murmur_data))
                              + len(os.listdir(extrastole_data))
                              + len(os.listdir(artifact_data))
                              + len(os.listdir(extrahls_data)))

print("No. of Sample for Test sounds: ", len(os.listdir(unlabel_data)))


x = np.array([len(os.listdir(normal_data)),
              len(os.listdir(murmur_data)),
              len(os.listdir(extrastole_data)),
              len(os.listdir(artifact_data)),
              len(os.listdir(extrahls_data))])
labels = ['normal', 'murmur', 'extrastole', 'artifact', 'extrahls']
plt.pie(x, labels = labels, autopct = '%.0f%%', radius= 1.5, textprops={'fontsize': 16})
plt.show()


# Listen to rondom audio from specific class
def random_sound (audio_class):
    random_sound = np.random.randint(0,len(os.listdir(audio_class))) 
    sound = os.listdir(audio_class)[random_sound]
    sound = audio_class+sound
    sound,sample_rate = librosa.load(sound)
    return ipd.Audio(sound,rate=sample_rate),sound

# show waveform of audio from dataset 
# X axis, represents time.
# Y-axis measures displacement of air molecules.
# This is where amplitude comes in. It measures how much a molecule is displaced from its resting position.  
import librosa.display
import matplotlib.pyplot as plt

# Function to show waveform of an audio sample
def show_audio_waveform(audio_sample, sr=22050):
    plt.figure(figsize=(20, 5))
    librosa.display.waveshow(audio_sample, sr=sr)  # Use waveshow instead of waveplot
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()
    
# show spectrum of audio from dataset 
def show_audio_spectrum(audio_sample):
    sample_rate = 22050
    fft_normal = np.fft.fft(audio_sample)
    magnitude_normal = np.abs(fft_normal)
    freq_normal = np.linspace(0,sample_rate, len(magnitude_normal)) 
    half_freq = freq_normal[:int(len(freq_normal)/2)]
    half_magnitude = magnitude_normal[:int(len(freq_normal)/2)]

    plt.figure(figsize=(12,8))
    plt.plot(half_freq,half_magnitude)
    plt.title("Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

# show specrogram of audio from dataset 
# the output is an image that represents a sound. 
# X-axis is for time, y-axis is for frequency and the color is for intensity
def show_spectrogram (audio_sample):    
    # STFT -> spectrogram
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples
    sample_rate = 22050

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate

    print("STFT hop length duration is: {}s".format(hop_length_duration))
    print("STFT window duration is: {}s".format(n_fft_duration))

    # perform stft
    stft_normal = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft_normal)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    # display spectrogram
    plt.figure(figsize=(15,10))
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    #plt.set_cmap("YlOrBr")
    plt.title("Spectrogram")

# MFCCs
# extract 52 MFCCs
# Function to show MFCC features of an audio sample
def show_mfcc_features(audio_sample, sr=22050, n_mfcc=52):
    # Parameters for MFCC computation
    n_fft = 2048
    hop_length = 512
    
    # Compute MFCC features
    MFCCs = librosa.feature.mfcc(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    
    # Display the MFCC features
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.title("MFCC")
    plt.show()
normal_audio, normal_sample  = random_sound(normal_data)
normal_audio

show_audio_waveform(normal_sample)
show_audio_spectrum(normal_sample)
show_spectrogram(normal_sample)
show_mfcc_features(normal_sample)

def add_noise(data,x):
    noise = np.random.randn(len(data))
    data_noise = data + x * noise
    return data_noise

def shift(data,x):
    return np.roll(data, x)

def stretch(data, rate):
    data = librosa.effects.time_stretch(data, rate)
    return data

def pitch_shift (data , rate):
    data = librosa.effects.pitch_shift(data, sr=220250, n_steps=rate)
    return data


def stretch(sound_data, rate):
    """
    Time-stretch the sound data by the given rate.
    """
    return librosa.effects.time_stretch(y=sound_data, rate=rate)

def load_file_data(folder, file_names, duration=10, sr=22050):
    '''
    Extract MFCC features from the audio data.
    Augment sound data by adding stretching.
    52 features are extracted from each audio file and used to train the model.
    '''
    input_length = sr * duration
    features = 52
    data = []

    for file_name in file_names:
        try:
            sound_file = os.path.join(folder, file_name)
            X, sr = librosa.load(sound_file, sr=sr, duration=duration)
            
            if X is None or len(X) == 0:
                raise ValueError(f"Audio data is empty or not loaded properly: {file_name}")
            
            dur = librosa.get_duration(y=X, sr=sr)

            # Pad audio file to ensure it has the same duration
            if round(dur) < duration:
                print(f"Fixing audio length: {file_name}")
                X = librosa.util.fix_length(X, size=input_length)

            # Extract normalized MFCC features from the data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=features).T, axis=0)
            data.append(mfccs)

            # Data augmentation - stretching
            stretch_data_1 = stretch(X, 0.8)
            mfccs_stretch_1 = np.mean(librosa.feature.mfcc(y=stretch_data_1, sr=sr, n_mfcc=features).T, axis=0)
            data.append(mfccs_stretch_1)

            stretch_data_2 = stretch(X, 1.2)
            mfccs_stretch_2 = np.mean(librosa.feature.mfcc(y=stretch_data_2, sr=sr, n_mfcc=features).T, axis=0)
            data.append(mfccs_stretch_2)

        except Exception as e:
            print(f"Error encountered while parsing file: {file_name}, error: {e}")

# Convert the list to a numpy array
    return np.array(data)


# Map integer value to text labels
CLASSES = ['artifact', 'murmur', 'normal','extrahls','extrastole']
NB_CLASSES = len(CLASSES)

label_to_int = {k: v for v, k in enumerate(CLASSES)}
print(label_to_int)
print(" ")
int_to_label = {v: k for k, v in label_to_int.items()}
print(int_to_label)

# Constants
SAMPLE_RATE = 22050
MAX_SOUND_CLIP_DURATION = 10

# Loading and labeling the data
artifact_files = fnmatch.filter(os.listdir(artifact_data), 'artifact*.wav')
artifact_sounds = load_file_data(folder=artifact_data, file_names=artifact_files, duration=MAX_SOUND_CLIP_DURATION)
artifact_labels = [0 for _ in artifact_sounds]

normal_files = fnmatch.filter(os.listdir(normal_data), 'normal*.wav')
normal_sounds = load_file_data(folder=normal_data, file_names=normal_files, duration=MAX_SOUND_CLIP_DURATION)
normal_labels = [2 for _ in normal_sounds]

extrahls_files = fnmatch.filter(os.listdir(extrahls_data), 'extrahls*.wav')
extrahls_sounds = load_file_data(folder=extrahls_data, file_names=extrahls_files, duration=MAX_SOUND_CLIP_DURATION)
extrahls_labels = [2 for _ in extrahls_sounds]

murmur_files = fnmatch.filter(os.listdir(murmur_data), 'murmur*.wav')
murmur_sounds = load_file_data(folder=murmur_data, file_names=murmur_files, duration=MAX_SOUND_CLIP_DURATION)
murmur_labels = [1 for _ in murmur_sounds]

extrastole_files = fnmatch.filter(os.listdir(extrastole_data), 'extrastole*.wav')
extrastole_sounds = load_file_data(folder=extrastole_data, file_names=extrastole_files, duration=MAX_SOUND_CLIP_DURATION)
extrastole_labels = [2 for _ in extrastole_sounds]

print("Loading Done")


# Assign distinct labels for each class

artifact_labels = [0 for _ in artifact_sounds]    # Artifact: 0
murmur_labels = [1 for _ in murmur_sounds]        # Murmur: 1
normal_labels = [2 for _ in normal_sounds]        # Normal: 2
extrahls_labels = [3 for _ in extrahls_sounds]    # Extrahls: 3
extrastole_labels = [4 for _ in extrastole_sounds]  # Extrastole: 4'''


# unlabel_data files conversion
Bunlabelledtest_files = fnmatch.filter(os.listdir(unlabel_data), 'Bunlabelledtest*.wav')
Bunlabelledtest_sounds = load_file_data(folder=unlabel_data,file_names=Bunlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)
Bunlabelledtest_labels = [-1 for items in Bunlabelledtest_sounds]

Aunlabelledtest_files = fnmatch.filter(os.listdir(unlabel_data), 'Aunlabelledtest*.wav')
Aunlabelledtest_sounds = load_file_data(folder=unlabel_data,file_names=Aunlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)
Aunlabelledtest_labels = [-1 for items in Aunlabelledtest_sounds]

print ("Loading of unlabel data done")

'''# Number of unlabelled files in each group
print(f"Number of Aunlabelledtest_files: {len(Aunlabelledtest_files)}")
print(f"Number of Bunlabelledtest_files: {len(Bunlabelledtest_files)}")

# Number of files combined
total_unlabelled_files = len(Aunlabelledtest_files) + len(Bunlabelledtest_files)
print(f"Total number of unlabelled files: {total_unlabelled_files}")

# Total unlabelled sounds and labels
total_unlabelled_sounds = len(Aunlabelledtest_sounds) + len(Bunlabelledtest_sounds)
total_unlabelled_labels = len(Aunlabelledtest_labels) + len(Bunlabelledtest_labels)

print(f"Total unlabelled sounds: {total_unlabelled_sounds}")
print(f"Total unlabelled labels: {total_unlabelled_labels}")'''

#combine set-a and set-b 
x_data = np.concatenate((artifact_sounds, normal_sounds,extrahls_sounds,murmur_sounds,extrastole_sounds))
print(x_data)
y_data = np.concatenate((artifact_labels, normal_labels,extrahls_labels,murmur_labels,extrastole_labels))

test_x = np.concatenate((Aunlabelledtest_sounds,Bunlabelledtest_sounds))
test_y = np.concatenate((Aunlabelledtest_labels,Bunlabelledtest_labels))

'''print ("combined training data record: ",len(x_data), len(test_x))


print("Unique classes in y_data:", np.unique(y_data))
print("Class counts in y_data:", np.bincount(y_data))'''

# Check the unique labels assigned to each category
print("Artifact labels unique values:", np.unique(artifact_labels))  # Should be [0]
print("Normal labels unique values:", np.unique(normal_labels))      # Should be [1 or 2]
print("Extrahls labels unique values:", np.unique(extrahls_labels))  # Should be [1 or 2]
print("Murmur labels unique values:", np.unique(murmur_labels))      # Should be [3]
print("Extrastole labels unique values:", np.unique(extrastole_labels))  # Should be [4]

    
    # split data into Train, Validation and Test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=42, shuffle=True)

# Classification Models
svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(x_train, y_train)

ada_classifier = AdaBoostClassifier(n_estimators=50)
ada_classifier.fit(x_train, y_train)

xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train)

# Regression Models
rf_regressor = RandomForestRegressor(n_estimators=100)
rf_regressor.fit(x_train, y_train)

logistic_regressor = LogisticRegression(max_iter=1000)
logistic_regressor.fit(x_train, y_train)

svr_regressor = SVR(kernel='linear')
svr_regressor.fit(x_train, y_train)

ada_regressor = AdaBoostRegressor(n_estimators=50)
ada_regressor.fit(x_train, y_train)

xgb_regressor = XGBRegressor()
xgb_regressor.fit(x_train, y_train)

# For classification evaluation
y_pred_svm = svm_model.predict(x_test)
y_pred_rf = rf_classifier.predict(x_test)
y_pred_ada = ada_classifier.predict(x_test)
y_pred_xgb = xgb_classifier.predict(x_test)

print(f"SVM Classification Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print(f"Random Forest Classification Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"AdaBoost Classification Accuracy: {accuracy_score(y_test, y_pred_ada)}")
print(f"XGBoost Classification Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

# For regression evaluation
y_pred_rf_reg = rf_regressor.predict(x_test)
y_pred_log_reg = logistic_regressor.predict(x_test)
y_pred_svr = svr_regressor.predict(x_test)
y_pred_ada_reg = ada_regressor.predict(x_test)
y_pred_xgb_reg = xgb_regressor.predict(x_test)

print(f"Random Forest Regression MAE: {mean_absolute_error(y_test, y_pred_rf_reg)}")
print(f"Logistic Regression MAE: {mean_absolute_error(y_test, y_pred_log_reg)}")
print(f"SVR Regression MAE: {mean_absolute_error(y_test, y_pred_svr)}")
print(f"AdaBoost Regression MAE: {mean_absolute_error(y_test, y_pred_ada_reg)}")
print(f"XGBoost Regression MAE: {mean_absolute_error(y_test, y_pred_xgb_reg)}")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

# --- CLASSIFICATION EVALUATION ---
classification_models = {
    "SVM_5": svm_model,
    "Random Forest_5": rf_classifier,
    "AdaBoost_5": ada_classifier,
    "XGBoost_5": xgb_classifier
}

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

for name, model in classification_models.items():
    # Training set evaluation
    y_train_pred = model.predict(x_train)
    y_train_pred_proba = model.predict_proba(x_train) if hasattr(model, "predict_proba") else None
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Handle multi-class AUC-ROC
    if y_train_pred_proba is not None:
        train_auc_roc = roc_auc_score(y_train, y_train_pred_proba, multi_class='ovr', average='weighted')
    else:
        train_auc_roc = "N/A"
    
    print(f"{name} - Training Set")
    print(f"Accuracy: {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall: {train_recall}")
    print(f"F1-Score: {train_f1}")
    print(f"AUC-ROC: {train_auc_roc}\n")
    
    # Test set evaluation
    y_test_pred = model.predict(x_test)
    y_test_pred_proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Handle multi-class AUC-ROC
    if y_test_pred_proba is not None:
        test_auc_roc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr', average='weighted')
    else:
        test_auc_roc = "N/A"
    
    print(f"{name} - Test Set")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1-Score: {test_f1}")
    print(f"AUC-ROC: {test_auc_roc}\n")


# --- REGRESSION EVALUATION ---
regression_models = {
    "Random Forest Regressor": rf_regressor,
    "Logistic Regressor": logistic_regressor,
    "SVR Regressor": svr_regressor,
    "AdaBoost Regressor": ada_regressor,
    "XGBoost Regressor": xgb_regressor
}

for name, model in regression_models.items():
    # Training set evaluation
    y_train_pred = model.predict(x_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print(f"{name} - Training Set")
    print(f"MAE: {train_mae}")
    print(f"RMSE: {train_rmse}\n")

    # Test set evaluation
    y_test_pred = model.predict(x_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"{name} - Test Set")
    print(f"MAE: {test_mae}")
    print(f"RMSE: {test_rmse}\n")

# Save models
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(rf_classifier, 'rf_classifier.pkl')
joblib.dump(ada_classifier, 'ada_classifier.pkl')
joblib.dump(xgb_classifier, 'xgb_classifier.pkl')

joblib.dump(rf_regressor, 'rf_regressor.pkl')
joblib.dump(logistic_regressor, 'logistic_regressor.pkl')
joblib.dump(svr_regressor, 'svr_regressor.pkl')
joblib.dump(ada_regressor, 'ada_regressor.pkl')
joblib.dump(xgb_regressor, 'xgb_regressor.pkl')

import joblib
import librosa
import os
import numpy as np

def load_audio_files_from_folder(folder_path, duration=10, sr=22050):
    """
    Load all audio files from a folder, extract MFCC features, and prepare them for prediction.
    """
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.wav')]
    features = []
    
    for file_path in file_paths:
        X, sr = librosa.load(file_path, sr=sr, duration=duration)
        dur = librosa.get_duration(y=X, sr=sr)
        
        # Pad audio file to the same duration
        if round(dur) < duration:
            X = librosa.util.fix_length(X, size=sr * duration)
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=52, n_fft=512, hop_length=2048).T, axis=0)
        features.append(mfccs)
    
    return file_paths, np.array(features)

def predict_with_all_models(file_paths, features, models, regression_models, classes=["artifact", "murmur", "normal", "extahls", "extrastole"]):
    """
    Predict using all classification models and regression models, and save the results to separate files.
    """
    for model_name, model in models.items():
        predictions = model.predict(features)
        results = []
        for i, file_path in enumerate(file_paths):
            predicted_class = classes[int(predictions[i])]
            results.append((file_path, predicted_class))
        
        # Save the results to a file
        with open(f'{model_name}_classification_results.txt', 'w') as file:
            for file_path, predicted_class in results:
                file.write(f"File: {file_path} | {model_name} Predicted Class: {predicted_class}\n")
    
    for reg_model_name, reg_model in regression_models.items():
        reg_predictions = reg_model.predict(features)
        reg_results = []
        for i, file_path in enumerate(file_paths):
            predicted_value = reg_predictions[i]
            reg_results.append((file_path, predicted_value))
        
        # Save regression results to a file
        with open(f'{reg_model_name}_regression_results.txt', 'w') as file:
            for file_path, predicted_value in reg_results:
                file.write(f"File: {file_path} | {reg_model_name} Regression Predicted Value: {predicted_value}\n")
    
    print("Predictions completed and results saved to respective files.")

# Load all trained classification models
svm_model = joblib.load('svm_model.pkl')
rf_classifier = joblib.load('rf_classifier.pkl')
ada_classifier = joblib.load('ada_classifier.pkl')
xgb_classifier = joblib.load('xgb_classifier.pkl')

# Load all trained regression models
rf_regressor = joblib.load('rf_regressor.pkl')
logistic_regressor = joblib.load('logistic_regressor.pkl')
svr_regressor = joblib.load('svr_regressor.pkl')
ada_regressor = joblib.load('ada_regressor.pkl')
xgb_regressor = joblib.load('xgb_regressor.pkl')

# Dictionary of classification models
classification_models = {
    'SVM': svm_model,
    'RandomForest': rf_classifier,
    'AdaBoost': ada_classifier,
    'XGBoost': xgb_classifier
}

# Dictionary of regression models
regression_models = {
    'RandomForest': rf_regressor,
    'LogisticRegression': logistic_regressor,
    'SVR': svr_regressor,
    'AdaBoost': ada_regressor,
    'XGBoost': xgb_regressor
}

# Load all audio files from the folder and extract features
folder_path = "/Unlabelled"
file_paths, features = load_audio_files_from_folder(folder_path)

# Make predictions with all models and save results
predict_with_all_models(file_paths, features, classification_models, regression_models)


def load_audio_files_from_folder(folder_path, duration=10, sr=22050):
    """
    Load all audio files from a folder, extract MFCC features, and prepare them for prediction.
    """
    file_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.wav')]
    features = []
    
    for file_path in file_paths:
        X, sr = librosa.load(file_path, sr=sr, duration=duration)
        dur = librosa.get_duration(y=X, sr=sr)
        
        # Pad audio file to the same duration
        if round(dur) < duration:
            X = librosa.util.fix_length(X, size=sr * duration)
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=52, n_fft=512, hop_length=2048).T, axis=0)
        features.append(mfccs)
    
    return file_paths, np.array(features)

def predict_with_models_combined(file_paths, features, classification_model, regression_model, model_name, classes=["artifact", "murmur", "normal","extrahls","extrastole"]):
    """
    Predict using both classification and regression models and save the combined results to one file.
    """
    # Classification Predictions
    class_predictions = classification_model.predict(features)
    
    # Regression Predictions
    regression_predictions = regression_model.predict(features)
    
    # Save combined results
    combined_results = []
    for i, file_path in enumerate(file_paths):
        predicted_class = classes[int(class_predictions[i])]
        predicted_value = regression_predictions[i]
        combined_results.append((file_path, predicted_class, predicted_value))
    
    # Save combined results to a single file
    with open(f'{model_name}_combined_results.txt', 'w') as file:
        file.write(f"File Path\tPredicted Class\tPredicted Value\n")
        for file_path, predicted_class, predicted_value in combined_results:
            file.write(f"{file_path}\t{predicted_class}\t{predicted_value}\n")
    
    print(f"Combined classification and regression results saved for {model_name}.")

# Load all trained classification models
svm_model = joblib.load('svm_model.pkl')
rf_classifier = joblib.load('rf_classifier.pkl')
ada_classifier = joblib.load('ada_classifier.pkl')
xgb_classifier = joblib.load('xgb_classifier.pkl')

# Load all trained regression models
rf_regressor = joblib.load('rf_regressor.pkl')
logistic_regressor = joblib.load('logistic_regressor.pkl')
svr_regressor = joblib.load('svr_regressor.pkl')
ada_regressor = joblib.load('ada_regressor.pkl')
xgb_regressor = joblib.load('xgb_regressor.pkl')

# Dictionary of classification models and their corresponding regression models
model_pairs = {
    'SVM': (svm_model, rf_regressor),
    'RandomForest': (rf_classifier, rf_regressor),
    'AdaBoost': (ada_classifier, ada_regressor),
    'XGBoost': (xgb_classifier, xgb_regressor)
}

# Load all audio files from the folder and extract features
folder_path = "/Unlabelled"
file_paths, features = load_audio_files_from_folder(folder_path)

# Make predictions with each model pair (classification + regression) and save combined results
for model_name, (classification_model, regression_model) in model_pairs.items():
    predict_with_models_combined(file_paths, features, classification_model, regression_model, model_name)


### Preparation for LSTM model
# One-Hot Encoding of labels
y_train = np.array(tf.keras.utils.to_categorical(y_train, len(CLASSES)))
y_test = np.array(tf.keras.utils.to_categorical(y_test, len(CLASSES)))
y_val = np.array(tf.keras.utils.to_categorical(y_val, len(CLASSES)))
test_y = np.array(tf.keras.utils.to_categorical(test_y, len(CLASSES)))

# Print the shape of data to verify
print(f"Train data shape: {x_train.shape}, Train labels shape: {y_train.shape}")
print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

# Prepare LSTM data
x_train_lstm = x_train
x_val_lstm = x_val
x_test_lstm = x_test

y_train_lstm = y_train  # Using one-hot encoded labels
y_val_lstm = y_val      # Using one-hot encoded labels
y_test_lstm = y_test    # Using one-hot encoded labels

# Build LSTM model
lstm_model = Sequential()

lstm_model.add(Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(52, 1)))
lstm_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
lstm_model.add(BatchNormalization())

lstm_model.add(Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
lstm_model.add(BatchNormalization())

lstm_model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
lstm_model.add(BatchNormalization())

lstm_model.add(LSTM(128, return_sequences=True))
lstm_model.add(LSTM(128))

lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dropout(0.3))

lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.3))

lstm_model.add(Dense(5, activation='softmax'))  # 5-class classification problem

# Model summary
lstm_model.summary()

# Compile the model
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
lstm_model.compile(optimizer=optimiser,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

# Callbacks: EarlyStopping and ModelCheckpoint
cb = [EarlyStopping(patience=30, monitor='val_accuracy', mode='max', restore_best_weights=True),
      ModelCheckpoint("Heart_LSTM.keras", save_best_only=True)]

# Train the model using one-hot encoded labels
history = lstm_model.fit(x_train_lstm, y_train_lstm,  # Using one-hot encoded labels for training
                         validation_data=(x_val_lstm, y_val_lstm),  # Using one-hot encoded labels for validation
                         batch_size=8, epochs=150,
                         callbacks=cb)

# Evaluate the model using validation data
lstm_model.evaluate(x_val_lstm, y_val_lstm)


def plot_loss_curves(history):
    
  """
  Returns separate loss curves for training and validation metrics.
  
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))
  # Plot accuracy
  plt.figure()
  plt.grid()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

plot_loss_curves(history)

classes = ["artifact" ,"murmur ", "normal", "extrahls","extrastole"]

preds = lstm_model.predict(x_test_lstm)
classpreds = [ np.argmax(t) for t in preds ]
y_testclass = [np.argmax(t) for t in y_test_lstm]
cm = confusion_matrix(y_testclass, classpreds)

plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', xticklabels=classes, yticklabels=classes)

plt.title('')
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.show(ax)

print(classification_report(y_testclass, classpreds, target_names=classes))


def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Extract features for unlabelled data
unlabel_files = os.listdir(unlabel_data)
X_unlabelled = []

for file in unlabel_files:
    file_path = os.path.join(unlabel_data, file)
    features = extract_features(file_path)
    X_unlabelled.append(features)

X_unlabelled = np.array(X_unlabelled)

# Reshape the data for LSTM input (LSTM expects 3D data)
X_unlabelled = X_unlabelled.reshape((X_unlabelled.shape[0], X_unlabelled.shape[1], 1))

# ## 4. LSTM Model Prediction

# Assuming the `lstm_model` is already loaded or trained before this section
lstm_model = load_model('/Heart_LSTM.keras')  # If the model is already saved, load it here

# Make predictions on the unlabelled data
lstm_predictions = lstm_model.predict(X_unlabelled)

# Convert predictions from one-hot encoding to label indices
lstm_predictions_labels = np.argmax(lstm_predictions, axis=1)

# Map label indices back to original labels (assuming you have a LabelEncoder)
label_encoder = preprocessing.LabelEncoder()
label_encoder.classes_ = np.array(['normal', 'murmur', 'artifact','extrahls','extrastole'])  # Original class names

# Convert numerical predictions back to string labels
predicted_labels = label_encoder.inverse_transform(lstm_predictions_labels)

# ## 5. Save Predictions to a File

output_df = pd.DataFrame({
    'File': unlabel_files,
    'Prediction': predicted_labels
})

output_df.to_csv('lstm_predictions_on_unlabelled_data.csv', index=False)

print("Predictions saved to lstm_predictions_on_unlabelled_data.csv")
