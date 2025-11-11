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

#%% ── MEMMAP SETUP FOR LARGE VCTK ────────────────────────────────────────────────
# (replaces the in-memory Low_STFT_FEATURES accumulation)
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
target_folder = r'E:\Raw Data\VCTK_16kHz\train\clean'

# 1) gather all wav paths
frame_length = int(0.032 * 16000)
wav_paths = []
for root, _, files in os.walk(target_folder):
    for f in files:
        if f.lower().endswith('.wav'):
            wav_paths.append(os.path.join(root, f))

# 2) count frames per file (pad to full frames)
file_frame_counts = []
for fp in wav_paths:
    sig, sr = load_audio(fp)
    if sig is None:
        file_frame_counts.append(0)
        continue
    pad = (frame_length - len(sig) % frame_length) % frame_length
    sig = np.pad(sig, (0, pad), 'constant')
    mag, _ = extract_stft(sig, sr, n_fft=n_fft, hop_length=hop_length)
    file_frame_counts.append(mag.shape[1])

total_frames = sum(file_frame_counts)
n_bins = n_fft // 2 + 1  # STFT bins

# 3) create memmaps
low_mm  = np.memmap('E:\low_stft.dat',  dtype='float32', mode='w+', shape=(total_frames, n_bins))
high_mm = np.memmap('E:\high_stft.dat', dtype='float32', mode='w+', shape=(total_frames, n_bins))

# 4) second pass: fill memmaps
write_idx = 0
for fp, n in zip(wav_paths, file_frame_counts):
    sig, sr = load_audio(fp)
    if sig is None or n == 0:
        continue
    pad = (frame_length - len(sig) % frame_length) % frame_length
    sig = np.pad(sig, (0, pad), 'constant')

    # highband STFT
    high_mag, _ = extract_stft(sig, sr, n_fft=n_fft, hop_length=hop_length)
    # lowband STFT
    low_sig = lowpass_filter(sig, cutoff=4000, fs=sr)
    low_mag, _ = extract_stft(low_sig, sr, n_fft=n_fft, hop_length=hop_length)

    # transpose to (frames, bins)
    high_mm[write_idx:write_idx+n, :] = high_mag.T
    low_mm [write_idx:write_idx+n, :] = low_mag.T
    write_idx += n

low_mm.flush()
high_mm.flush()

# 5) fit & save scalers (once)
from sklearn.preprocessing import StandardScaler
import joblib

feature_scaler = StandardScaler().fit(low_mm)
label_scaler   = StandardScaler().fit(high_mm)
joblib.dump(feature_scaler, 'feature_scaler_vctk.pkl')
joblib.dump(label_scaler,   'label_scaler_vctk.pkl')

#%% ── LOAD SCALERS & SPLIT INTO TRAIN/VAL/TEST ─────────────────────────────────

feature_scaler = joblib.load('feature_scaler_vctk.pkl')
label_scaler   = joblib.load('label_scaler_vctk.pkl')
mean_lo, scale_lo = feature_scaler.mean_, feature_scaler.scale_
mean_hi, scale_hi = label_scaler.mean_,   label_scaler.scale_

# build index array and split
import numpy as np
from sklearn.model_selection import train_test_split

indices = np.arange(total_frames)
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# normalize & slice memmaps
X_train = (low_mm [train_idx] - mean_lo) / scale_lo
y_train = (high_mm[train_idx] - mean_hi) / scale_hi
X_val   = (low_mm [val_idx]   - mean_lo) / scale_lo
y_val   = (high_mm[val_idx]   - mean_hi) / scale_hi
X_test  = (low_mm [test_idx]  - mean_lo) / scale_lo
y_test  = (high_mm[test_idx]  - mean_hi) / scale_hi

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
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val)
)
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
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

model.save('Model_VCTK.h5')