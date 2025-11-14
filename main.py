import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import math
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt
from pesq import pesq
from pystoi import stoi
import joblib

import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs visible:   ", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def plot_spectrogram(signal, sample_rate, title='Spectrogram'):
   D = librosa.amplitude_to_db(np.abs(librosa.stft(signal,n_fft=1024, hop_length=128)), ref=np.max)
   plt.figure(figsize=(10, 3))
   librosa.display.specshow(D, sr=sample_rate, hop_length=128, x_axis='time', y_axis='hz')
   plt.colorbar(format='%+2.0f dB')
   plt.title(title)
   plt.ylabel(0-16000)
   plt.ylabel('Hz')
   plt.show()

def apply_fft(signal, fs,a):
   fft_signal = np.fft.fft(signal)
   n = len(signal)
   freq = np.fft.fftfreq(n, 1/fs)
   plt.figure(figsize=(10, 3))
   plt.plot(freq, np.abs(fft_signal))# Plot both frequencies
   plt.title(f'FFT Magnitude of {a}')
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Magnitude')
   plt.grid(True)
   plt.show()

def SNR(r, t, skip=8192):
   difference = ((r[skip: -skip] - t[skip: -skip]) ** 2).sum()
   return lin2db((r[skip: -skip] ** 2).sum() / difference)

def UpsampledAudio(audio,fs):
   audio_resampled = librosa.resample(audio, orig_sr=fs, target_sr=8000)
   audio_upsampled = librosa.resample(audio_resampled, orig_sr=8000, target_sr=16000)
   return audio_upsampled

def lowpass_filter(data, cutoff, fs=16000, order=50):
   nyquist = fs / 2
   #print(cutoff)
   normalized_cutoff = cutoff / nyquist
   sos = scipy.signal.butter(order, normalized_cutoff, btype='low', output='sos')
   signal_prefiltered = scipy.signal.sosfilt(sos, data)
   return signal_prefiltered

def extract_stft(frame, sample_rate, n_fft=670,hop_length=335, win_length=670):
    if win_length is None:
        win_length = n_fft
    stft_complex = librosa.stft(frame, n_fft=n_fft, hop_length= hop_length, win_length=win_length, window= "hamming")

    # Compute magnitude and convert to decibels (dB). The dB conversion helps in dynamic range compression.
    # stft_mag = librosa.amplitude_to_db(np.abs(stft_complex), ref=np.max) #not sure bout this one
    mag = np.abs(stft_complex)
    mag = np.maximum(mag, 1e-8)
    stft_mag = 20 * np.log10(np.abs(mag))
    stft_phase = np.angle(stft_complex)

    ################no additional bincopy preperation here############################

    return stft_mag, stft_phase

def load_audio(file_path):
    try:
        # Attempt to load with soundfile
        audio_signal, fs = sf.read(file_path)
    except Exception as e:
        print(f"Soundfile failed: {e}")
        try:
            # Fallback to pydub
            audio = AudioSegment.from_file(file_path)

            audio_signal = np.array(audio.get_array_of_samples())
            fs = audio.frame_rate
        except Exception as e:
            print(f"Pydub also failed: {e}")
            return None, None
    return audio_signal, fs

#%%

#PARAMETERS
minW=512
min_freq=62
max_freq=7900
Binsperoctave=48
fs =16000
num_bins_additional = 0
binsrequired=0
Total_bins= 0
num_bins_low= 0
n_fft = 670
hop_length = 335


Low_STFT_FEATURES = []
High_STFT_FEATURES = []
Low_STFT_FEATURES_PHASE=[]
High_STFT_FEATURES_PHASE=[]

def create_dataset_stft(file_path):

   audio_signal, fs = load_audio(file_path)
   if audio_signal is None or fs is None:
        return
   frame_length = int(0.032 * fs)

   #padding the signal to make it equal to minimum window length
   pad_width = (frame_length - len(audio_signal) % frame_length) % frame_length
   audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

   # Highband STFT: full frequency range (up to Nyquist)
   high_stft_mag, high_stft_phase = extract_stft(audio_signal.astype(np.float32), fs, n_fft=n_fft,
                                                 hop_length=hop_length)
   High_STFT_FEATURES.extend(high_stft_mag.T)  # Transposing to match [frames, bins] if needed.
   High_STFT_FEATURES_PHASE.extend(high_stft_phase.T)

   # Lowband STFT: Apply a lowpass filter with a cutoff (e.g., 4000 Hz), then extract STFT features.
   low_audio = lowpass_filter(audio_signal, cutoff=4000, fs=fs)
   low_stft_mag, low_stft_phase = extract_stft(low_audio.astype(np.float32), fs, n_fft=n_fft, hop_length=hop_length)

   #################################################################################
   #bin copy logic completely deleted maybe consider to include this afterwards
   #################################################################################

   Low_STFT_FEATURES.extend(low_stft_mag.T)  # Adjust the shape accordingly.
   Low_STFT_FEATURES_PHASE.extend(low_stft_phase.T)


# Process the TIMIT dataset
target_folder = r'DSET\train\clean'


drfolders = os.listdir(target_folder)

file_limit = 10000000  # Set the limit to 1 file for testing
processed_files = 0
Timit = 0
for root, dirs, files in os.walk(target_folder):
   for file in files:
       if file.endswith('.wav'):
           file_pathlow = os.path.join(root, file)
           Timit += 1
           processed_files +=1
           # print(f'{Timit}{file_pathlow}')
           create_dataset_stft(file_pathlow)
           if processed_files >= file_limit:
                       break
   if processed_files >= file_limit:
           break
# print("Processing complete.")
# print("Timit files =", Timit)

#%%

def calculate_power(cqt_mag):
   return np.square(cqt_mag)

def scale_additional_low_bins(low_cqt, original_cqt, total_bins, min_bin):
   # Ensure inputs are numpy arrays
   low_cqt = np.array(low_cqt)
   original_cqt = np.array(original_cqt)

   # Calculate the power of the bins from min_bin to total_bins for the original CQT
   original_bins_power = calculate_power(original_cqt[:, min_bin:total_bins])

   # Compute the mean power of these bins
   original_bins_power_mean = np.mean(original_bins_power)
   print("original_bins_power_mean:", original_bins_power_mean)

   # Calculate the power of the corresponding bins in the low CQT
   low_cqt_bins_power = calculate_power(low_cqt[:, min_bin:total_bins])

   # Compute the mean power of these bins in the low CQT
   low_cqt_bins_power_mean = np.mean(low_cqt_bins_power)
   print("low_cqt_bins_power_mean:", low_cqt_bins_power_mean)

   # Determine the scaling factor to match the power
   scaling_factor = np.sqrt(original_bins_power_mean / low_cqt_bins_power_mean)
   print("scaling_factor:", scaling_factor)

   # Scale only the bins from min_bin to total_bins
   scaled_low_cqt = low_cqt.copy()  # Copy to avoid modifying the original data
   scaled_low_cqt[:, min_bin:total_bins] *= scaling_factor
   return scaled_low_cqt



# Ensure all lists are converted to numpy arrays

Low_STFT_FEATURES = np.array(Low_STFT_FEATURES)
High_STFT_FEATURES = np.array(High_STFT_FEATURES)
Low_STFT_FEATURES_PHASE = np.array(Low_STFT_FEATURES_PHASE)
High_STFT_FEATURES_PHASE = np.array(High_STFT_FEATURES_PHASE)

# # # Scale Low_CQT_FEATURES
# scaled_Low_CQT_FEATURES = scale_additional_low_bins(Low_CQT_FEATURES, High_CQT_FEATURES, Total_bins, num_bins_low)

#%%
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
feature_scaler.fit(Low_STFT_FEATURES)
X_train_normalized = feature_scaler.transform(Low_STFT_FEATURES)

# Fit the scaler on the training labels
label_scaler = StandardScaler()
label_scaler.fit(High_STFT_FEATURES)
y_train_normalized = label_scaler.transform(High_STFT_FEATURES)

joblib.dump(feature_scaler, 'feature_scaler.pkl')
joblib.dump(label_scaler,   'label_scaler.pkl')
print("✓  scalers saved to feature_scaler.pkl and label_scaler.pkl")

# here create phase as a 2nd channel
# X_train = np.stack((X_train_normalized, Low_CQT_FEATURES_PHASE), axis=-1)
# Y_label = np.stack((y_train_normalized, High_CQT_FEATURES_PHASE), axis=-1)
# Train-test split
Xq_train, Xq_temp, y_train, y_temp = train_test_split(X_train_normalized, y_train_normalized, test_size=0.2, random_state=42)
Xq_val, Xq_test, y_val, y_test = train_test_split(Xq_temp, y_temp, test_size=0.2, random_state=42)

# Converting to NumPy arrays (if not already)
Xq_train = np.array(Xq_train)
Xq_val = np.array(Xq_val)
Xq_test = np.array(Xq_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


# Xq_train = Xq_train[..., np.newaxis]
# Xq_val = Xq_val[..., np.newaxis]
# Xq_test = Xq_test[..., np.newaxis]
# y_train = y_train[..., np.newaxis]
# y_val = y_val[..., np.newaxis]
# y_test = y_test[..., np.newaxis]
# # Print shapes of splits
print(f"Xq_train shape: {Xq_train.shape}, y_train shape: {y_train.shape}")
print(f"Xq_val shape: {Xq_val.shape}, y_val shape: {y_val.shape}")
# print(f"Xq_test shape: {Xq_test.shape}, y_test shape: {y_test.shape}”)

#%% MLP Best performer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(336,))) #changed
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(336, activation='linear'))


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.summary()

# Compile the model with appropriate optimizer, loss, and metrics
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])


# Train the model
history = model.fit(Xq_train, y_train, epochs=50, batch_size=64, validation_data=(Xq_val, y_val))
# Extract the loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# Evaluate the model
test_loss, test_mae = model.evaluate(Xq_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

model.save('Model_L_1.h5')