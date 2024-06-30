from lib.readers import ARTFReader
from lib.hdf5_reader_module import SignalClass

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mdp
import pandas as pd
import shutil
import copy
from scipy import signal
from scipy.linalg import eigh

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# knn
from sklearn.neighbors import KNeighborsClassifier
# svc
from sklearn.svm import SVC
# random forest
from sklearn.ensemble import RandomForestClassifier
# gradient boosting
from sklearn.ensemble import GradientBoostingClassifier


WINDOW_SIZE_SEC = 10

HDF5_FILE = "data/2024-03-04/dataset_0/TBI_007_v3_1_2_17.hdf5"
ARTF_FILE = "data/2024-03-04/dataset_0/TBI_007_v3_1_2_17.artf"
ABP_KEY = "art"


def read_artf(filename, abp_name="abp"):
    """
    Read artefacts from file

    Returns:
        global_artefacts: list of global artefacts
        icp_artefacts: list of ICP artefacts
        abp_artefacts: list of ABP artefacts
        metadata: metadata object
    """
    reader = ARTFReader(filename)
    return reader.read(abp_name=abp_name)

def read_hdf5(filename, signal="icp"):
    """
    Read HDF5 file

    Returns:
        data: numpy array of data
        sample_rate: sample rate of data
        start_time_s: start time of data in seconds
        end_time_s: end time of data in seconds
    """
    hdf5_file = h5py.File(filename, 'r') 
    wave_data = SignalClass(hdf5_file, signal)

    start_time = wave_data.get_all_data_start_time()
    start_time_s = start_time / 1e6
    end_time = wave_data.get_all_data_end_time()
    end_time_s = end_time / 1e6
    stream_duration_microsec = end_time - start_time

    stream = wave_data.get_data_stream(start_time, stream_duration_microsec)
    data = np.array(stream.values)

    sample_rate = stream.sampling_frq

    return data, sample_rate, start_time_s, end_time_s

def butterworth_filter_icp(x, sr):
    # filter do 10 Hz

    b, a = signal.butter(4, 10 / (sr / 2), 'low')
    y = signal.filtfilt(b, a, x)

    return y

def butterworth_filter_abp(x, sr):
    # filter to 20 Hz

    b, a = signal.butter(4, 20 / (sr / 2), 'low')
    y = signal.filtfilt(b, a, x)

    return y

def compute_stft(X, sr):
    sr = int(sr)

    f, t, Zxx = signal.stft(X, sr, window="hamming", nperseg=500, noverlap=450
                            )
    return f, t, Zxx

def upsample_signal(X, sr, target_sr):
    """
    Upsample signal to target sample rate.
    """
    upsample_factor = int(target_sr / sr)

    # mirror signal in front and back to avoid edge effects
    X = np.concatenate([X[::-1], X, X[::-1]])

    # upsample 
    X_upsampled = signal.resample(X, len(X) * upsample_factor)

    # get middle part
    X_upsampled = X_upsampled[int(len(X_upsampled)/3):int(2*len(X_upsampled)/3)]
    return X_upsampled

def max_consecutive_nans(X):
    max_consecutive = 0
    consecutive = 0
    for i in range(len(X)):
        if np.isnan(X[i]):
            consecutive += 1
        else:
            if consecutive > max_consecutive:
                max_consecutive = consecutive
            consecutive = 0
    return max_consecutive


def interpolate_nans(X, n=100):
    x = X
    # mirror signal
    x = np.concatenate([x[::-1], x, x[::-1]])
    x = pd.Series(x)
    x = x.interpolate(method='quadratic')

    # extract middle part
    x = x[int(len(x)/3):int(2*len(x)/3)]
    return x.values


icp_data, icp_sr, icp_start_time_s, icp_end_time_s = read_hdf5(HDF5_FILE, signal="icp")
abp_data, abp_sr, abp_start_time_s, abp_end_time_s = read_hdf5(HDF5_FILE, signal=ABP_KEY)

icp_length_s = icp_end_time_s - icp_start_time_s
abp_length_s = abp_end_time_s - abp_start_time_s

icp_length_h = icp_length_s / 3600
abp_length_h = abp_length_s / 3600

print(f"ICP length: {icp_length_s} s ({icp_length_h} h)")
print(f"ABP length: {abp_length_s} s ({abp_length_h} h)")

global_artefacts, icp_artefacts, abp_artefacts, metadata = read_artf(ARTF_FILE, abp_name=ABP_KEY)
icp_artefacts = sorted(icp_artefacts + global_artefacts, key=lambda x: x.start_time.timestamp())
abp_artefacts = sorted(abp_artefacts + global_artefacts, key=lambda x: x.start_time.timestamp())

# get normal segments
#x_icp_normal = icp_data[(217) * (int(icp_sr) * WINDOW_SIZE_SEC):(218) * (int(icp_sr) * WINDOW_SIZE_SEC)]
x_icp_normal = icp_data[(1) * (int(icp_sr) * WINDOW_SIZE_SEC):(2) * (int(icp_sr) * WINDOW_SIZE_SEC)]
x_abp_normal = abp_data[1 * (int(abp_sr) * WINDOW_SIZE_SEC):2 * (int(abp_sr) * WINDOW_SIZE_SEC)]

x_icp_normal_nans_idx = np.isnan(x_icp_normal)

x_abp_normal_filtered = butterworth_filter_abp(x_abp_normal, abp_sr)
x_icp_normal_filtered = butterworth_filter_icp(x_icp_normal, icp_sr)

#### NaN handling 

plt.rcParams["figure.figsize"] = (10, 6)

plt.subplot(2, 2, 1)
x = x_abp_normal.copy()
nan_start = 600
nan_end = 650
x[nan_start:nan_end] = np.nan
x_int = interpolate_nans(x)
mse = np.square(np.subtract(x_abp_normal, x_int)).mean()
print(mse)
plt.title(f"ABP Interpolace NaN (MSE {mse:.2f})")
plt.plot(x_abp_normal, label="ABP")
plt.plot(x_int, label="ABP interpolace", color="tab:orange")
plt.axvspan(nan_start, nan_end, color='red', alpha=0.25)
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / abp_sr)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.ylabel("ABP (mmHg)")
plt.legend()

plt.subplot(2, 2, 3)
x = x_icp_normal.copy()
nan_start = 600
nan_end = 650
x[nan_start:nan_end] = np.nan
x_int = interpolate_nans(x)
mse = np.square(np.subtract(x_icp_normal, x_int)).mean()
print(mse)
plt.title(f"ICP Interpolace NaN (MSE {mse:.2f})")
plt.plot(x_icp_normal, label="ICP")
plt.plot(x_int, label="ICP interpolace", color="tab:orange")
plt.axvspan(nan_start, nan_end, color='red', alpha=0.25)
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / icp_sr)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.ylabel("ICP (mmHg)")
plt.legend()

plt.subplot(2, 2, 2)
x = x_abp_normal.copy()
nan_start = 550
nan_end = 600
x[nan_start:nan_end] = np.nan
x_int = interpolate_nans(x)
mse = np.square(np.subtract(x_abp_normal, x_int)).mean()
print(mse)
plt.title(f"ABP Interpolace NaN (MSE {mse:.2f})")
plt.plot(x_abp_normal, label="ABP")
plt.plot(x_int, label="ABP interpolace", color="tab:orange")
plt.axvspan(nan_start, nan_end, color='red', alpha=0.25)
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / abp_sr)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.ylabel("ABP (mmHg)")
plt.legend()

plt.subplot(2, 2, 4)
x = x_icp_normal.copy()
nan_start = 550
nan_end = 600
x[nan_start:nan_end] = np.nan
x_int = interpolate_nans(x)
mse = np.square(np.subtract(x_icp_normal, x_int)).mean()
print(mse)
plt.title(f"ICP Interpolace NaN (MSE {mse:.2f})")
plt.plot(x_icp_normal, label="ICP")
plt.plot(x_int, label="ICP interpolace", color="tab:orange")
plt.axvspan(nan_start, nan_end, color='red', alpha=0.25)
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / icp_sr)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.ylabel("ICP (mmHg)")
plt.legend()

plt.tight_layout()
plt.show()


###### NORMAL 

plt.rcParams["figure.figsize"] = (10, 6)

plt.subplot(2, 1, 1)
plt.plot(x_abp_normal, label="ABP")
plt.plot(x_abp_normal_filtered, label="ABP filtrované")
plt.legend()
plt.xlabel("Čas [s]")
plt.ylabel("ABP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / abp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)

plt.subplot(2, 1, 2)
plt.plot(x_icp_normal, label="ICP")
plt.plot(x_icp_normal_filtered, label="ICP filtrované")
# red highlight for NaNs using axvspan
for i in range(len(x_icp_normal_nans_idx)):
    if x_icp_normal_nans_idx[i]:
        plt.axvspan(i - 1, i, color='red', alpha=0.5)
plt.xlabel("Čas [s]")
plt.ylabel("ICP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / icp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.legend()

plt.tight_layout()
plt.show()

###### NORMAL SPECTROGRAM STFT

plt.rcParams["figure.figsize"] = (10, 6)

f, t, Zxx = signal.stft(x_abp_normal, abp_sr, window="hamming", nperseg=10, noverlap=5)

plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.colorbar(label='Amplituda')
plt.title("STFT ABP")
plt.xlabel("Čas [s]")
plt.ylabel("Frekvence [Hz]")
plt.ylim(0, 20)
plt.tight_layout()
plt.show()

###### ARTEFACTS

abp_artefact_start_idxs = []
for artefact in abp_artefacts:
    artefact_idx_start, artefact_idx_end = artefact.to_index(abp_sr, WINDOW_SIZE_SEC, abp_start_time_s)
    abp_artefact_start_idxs.append(artefact_idx_start)

icp_artefact_start_idxs = []
for artefact in icp_artefacts:
    artefact_idx_start, artefact_idx_end = artefact.to_index(icp_sr, WINDOW_SIZE_SEC, icp_start_time_s)
    icp_artefact_start_idxs.append(artefact_idx_start)

#for i, art in enumerate(abp_artefacts):
#    artefact_idx = art.to_index(abp_sr, WINDOW_SIZE_SEC, abp_start_time_s)
#    print(f"Artefakt {i}: {artefact_idx}")
#    x = abp_data[artefact_idx[0]:artefact_idx[1]]
#    plt.plot(x)
#    plt.title(f"Artefakt {i}")
#    plt.show()

icp_artefact_idx = icp_artefacts[1].to_index(icp_sr, WINDOW_SIZE_SEC, icp_start_time_s)
abp_artefact_idx = abp_artefacts[1].to_index(abp_sr, WINDOW_SIZE_SEC, abp_start_time_s)
print("ICP artefact index:", icp_artefact_idx, "ABP artefact index:", abp_artefact_idx)
x_icp_artefact = icp_data[icp_artefact_idx[0]:icp_artefact_idx[1]]
x_abp_artefact = abp_data[abp_artefact_idx[0]:abp_artefact_idx[1]]

x_icp_artefact_filtered = butterworth_filter_icp(x_icp_artefact, icp_sr)
x_abp_artefact_filtered = butterworth_filter_abp(x_abp_artefact, abp_sr)


###### ARTEFACT

plt.rcParams["figure.figsize"] = (10, 6)

plt.subplot(2, 1, 1)
plt.plot(x_abp_artefact, label="ABP")
plt.plot(x_abp_artefact_filtered, label="ABP filtrované")
plt.legend()
plt.xlabel("Čas [s]")
plt.ylabel("ABP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / abp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)

plt.subplot(2, 1, 2)
plt.plot(x_icp_artefact, label="ICP")
plt.plot(x_icp_artefact_filtered, label="ICP filtrované")
plt.xlabel("Čas [s]")
plt.ylabel("ICP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / icp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.legend()

plt.tight_layout()
plt.show()



###### FILTERED

plt.rcParams["figure.figsize"] = (10, 6)

plt.subplot(2, 2, 1)
mse = np.square(np.subtract(x_abp_normal, x_abp_normal_filtered)).mean()
plt.title(f"ABP filtrované (MSE {mse:.2f})")
plt.plot(x_abp_normal, label="ABP")
plt.plot(x_abp_normal_filtered, label="ABP filtrované")
plt.legend()
plt.xlabel("Čas [s]")
plt.ylabel("ABP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / abp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)

plt.subplot(2, 2, 3)
mse = np.square(np.subtract(x_icp_normal, x_icp_normal_filtered)).mean()
plt.title(f"ICP filtrované (MSE {mse:.2f})")
plt.plot(x_icp_normal, label="ICP")
plt.plot(x_icp_normal_filtered, label="ICP filtrované")
# red highlight for NaNs using axvspan
for i in range(len(x_icp_normal_nans_idx)):
    if x_icp_normal_nans_idx[i]:
        plt.axvspan(i - 1, i, color='red', alpha=0.5)
plt.xlabel("Čas [s]")
plt.ylabel("ICP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / icp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.legend()

plt.subplot(2, 2, 2)
mse = np.square(np.subtract(x_abp_artefact, x_abp_artefact_filtered)).mean()
plt.title(f"ABP filtrované (MSE {mse:.2f})")
plt.plot(x_abp_artefact, label="ABP")
plt.plot(x_abp_artefact_filtered, label="ABP filtrované")
plt.legend()
plt.xlabel("Čas [s]")
plt.ylabel("ABP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / abp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)

plt.subplot(2, 2, 4)
mse = np.square(np.subtract(x_icp_artefact, x_icp_artefact_filtered)).mean()
plt.title(f"ICP filtrované (MSE {mse:.2f})")
plt.plot(x_icp_artefact, label="ICP")
plt.plot(x_icp_artefact_filtered, label="ICP filtrované")
plt.xlabel("Čas [s]")
plt.ylabel("ICP (mmHg)")
xticks = plt.gca().get_xticks()
xticks_labels = [int(xtick / icp_sr) for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)
plt.legend()

plt.tight_layout()
plt.show()

###### Upsample

upsample_factor = 100


x_abp_normal_upsampled = upsample_signal(x_abp_normal, abp_sr, abp_sr * upsample_factor)
x_abp_normal_upsampled_sr = int(abp_sr * upsample_factor)

x_icp_normal_upsampled = upsample_signal(x_icp_normal, icp_sr, icp_sr * upsample_factor)
x_icp_normal_upsampled_sr = int(icp_sr * upsample_factor)

x_abp_artefact_upsampled = upsample_signal(x_abp_artefact, abp_sr, abp_sr * upsample_factor)
x_abp_artefact_upsampled_sr = int(abp_sr * upsample_factor)

x_icp_artefact_upsampled = upsample_signal(x_icp_artefact, icp_sr, icp_sr * upsample_factor)
x_icp_artefact_upsampled_sr = int(icp_sr * upsample_factor)

###### STFT

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
# stft
f, t, Zxx = compute_stft(x_abp_normal_upsampled, sr=x_abp_normal_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
print(Zxx.shape)

plt.imshow(Zxx, aspect='auto', origin='lower',
           extent=[t[0], t[-1], f[0], f[-1]],
           vmin=0, vmax=Zxx_max)
plt.ylim(0, 20)
plt.xlabel("Čas [s]")
plt.ylabel("Frekvence [Hz]")
plt.title("STFT ABP")


plt.subplot(2, 2, 3)
# STFT
f, t, Zxx = compute_stft(x_icp_normal_upsampled, sr=x_icp_normal_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
print(Zxx.shape)

plt.imshow(Zxx, aspect='auto', origin='lower',
              extent=[t[0], t[-1], f[0], f[-1]],
              vmin=0, vmax=Zxx_max)
plt.ylim(0, 10)
plt.xlabel("Čas [s]")
plt.ylabel("Frekvence [Hz]")
plt.title("STFT ICP")


plt.subplot(2, 2, 2)
# stft
f, t, Zxx = compute_stft(x_abp_artefact_upsampled, sr=x_abp_artefact_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)

plt.imshow(Zxx, aspect='auto', origin='lower',
              extent=[t[0], t[-1], f[0], f[-1]],
              vmin=0, vmax=Zxx_max)
plt.ylim(0, 20)
plt.xlabel("Čas [s]")
plt.ylabel("Frekvence [Hz]")
plt.title("STFT ABP anomálie")


plt.subplot(2, 2, 4)
# stft
f, t, Zxx = compute_stft(x_icp_artefact_upsampled, sr=x_icp_artefact_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)

plt.imshow(Zxx, aspect='auto', origin='lower',
                extent=[t[0], t[-1], f[0], f[-1]],
                vmin=0, vmax=Zxx_max)
plt.ylim(0, 10)
plt.xlabel("Čas [s]")
plt.ylabel("Frekvence [Hz]")
plt.title("STFT ICP anomálie")


plt.tight_layout()
plt.show()



###### STFT SFA

upsample_factor = 100


x_abp_normal_upsampled = upsample_signal(x_abp_normal, abp_sr, abp_sr * upsample_factor)
x_abp_normal_upsampled_sr = int(abp_sr * upsample_factor)

x_icp_normal_upsampled = upsample_signal(x_icp_normal, icp_sr, icp_sr * upsample_factor)
x_icp_normal_upsampled_sr = int(icp_sr * upsample_factor)

x_abp_artefact_upsampled = upsample_signal(x_abp_artefact, abp_sr, abp_sr * upsample_factor)
x_abp_artefact_upsampled_sr = int(abp_sr * upsample_factor)

x_icp_artefact_upsampled = upsample_signal(x_icp_artefact, icp_sr, icp_sr * upsample_factor)
x_icp_artefact_upsampled_sr = int(icp_sr * upsample_factor)

plt.rcParams["figure.figsize"] = (10, 6)

for i in range(8):
#for i in range(8, 14):
    #plt.subplot(3, 2, i-7)
    plt.subplot(4, 2, i+1)
    # stft sfa
    f, t, Zxx = compute_stft(x_abp_normal_upsampled, sr=x_abp_normal_upsampled_sr)
    Zxx_max = np.max(np.abs(Zxx))
    Zxx = np.abs(Zxx)
    x = np.abs(Zxx).T
    number_features = 14
    sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
    features = sfa.execute(x)
    features = features.T
    print(features.shape)

    plt.plot(features[i])
    plt.xlabel("Čas [s]")
    plt.ylabel("Amplituda")
    xticks = np.arange(0, features.shape[1], 200)
    xticks_labels = [int(xtick / 200) for xtick in xticks]
    plt.xticks(xticks, xticks_labels)
    plt.title(f"STFT SFA ABP vlastnost {i + 1}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
# stft sfa
f, t, Zxx = compute_stft(x_abp_normal_upsampled, sr=x_abp_normal_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
x = np.abs(Zxx).T
number_features = 14
sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
features = sfa.execute(x)
features = features.T
print(features.shape)

plt.imshow(features, aspect='auto', origin='lower',
           extent=[0, features.shape[1], 0, features.shape[0]],
           vmin=0, vmax=1)
#plt.ylim(0, 20)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("STFT SFA ABP")


plt.subplot(2, 2, 3)
# stft sfa
f, t, Zxx = compute_stft(x_icp_normal_upsampled, sr=x_icp_normal_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
x = np.abs(Zxx).T

sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
features = sfa.execute(x)
features = features.T
print(features.shape)

plt.imshow(features, aspect='auto', origin='lower',
                extent=[0, features.shape[1], 0, features.shape[0]],
                vmin=0, vmax=1)
#plt.ylim(0, 10)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("STFT SFA ICP")


plt.subplot(2, 2, 2)
# stft sfa
f, t, Zxx = compute_stft(x_abp_artefact_upsampled, sr=x_abp_artefact_upsampled_sr) 
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
x = np.abs(Zxx).T 

sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
features = sfa.execute(x)
features = features.T

plt.imshow(features, aspect='auto', origin='lower',
                extent=[0, features.shape[1], 0, features.shape[0]],
                vmin=0, vmax=1)
#plt.ylim(0, 20)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("STFT SFA ABP anomálie")


plt.subplot(2, 2, 4)
# stft sfa
f, t, Zxx = compute_stft(x_icp_artefact_upsampled, sr=x_icp_artefact_upsampled_sr)
Zxx_max = np.max(np.abs(Zxx))
Zxx = np.abs(Zxx)
x = np.abs(Zxx).T 

sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
features = sfa.execute(x)
features = features.T

plt.imshow(features, aspect='auto', origin='lower',
                extent=[0, features.shape[1], 0, features.shape[0]],
                vmin=0, vmax=1)
#plt.ylim(0, 10)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("STFT SFA ICP anomálie")


plt.tight_layout()
plt.show()



##### detection

def process_segment(X, sr, is_icp: bool):

    consecutive_nans = max_consecutive_nans(X)
    if consecutive_nans > 50:
        raise ValueError("Too many consecutive NaNs")

    if np.sum(np.isnan(X)) > 0:
        X = interpolate_nans(X)
        #assert np.sum(np.isnan(X)) == 0, "NaNs in segment"
        if np.sum(np.isnan(X)) > 0:
            raise ValueError("NaNs in segment")
        #raise ValueError("NaNs in segment")

    # filter
    X_filtered = butterworth_filter_icp(X, sr) if is_icp else butterworth_filter_abp(X, sr)

    # upsample
    X_upsampled = upsample_signal(X, sr, 10000)

    # compute STFT
    f, t, Zxx = compute_stft(X_upsampled, sr=10000)

    # compute SFA
    x = np.abs(Zxx).T
    number_features = 14
    sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
    slow_features = sfa.execute(x)
    if slow_features.shape[1] != number_features:
        raise ValueError("Not enough features")

    return slow_features

def extract_features(slow_features):
    slow_features = slow_features.T

    # split into windows
    slow_features = np.array_split(slow_features.T, 10)
    features = []
    for sf in slow_features:
        sf = sf.T
        features.append([
            np.mean(sf, axis=1),
            np.std(sf, axis=1),
            np.min(sf, axis=1),
            np.max(sf, axis=1),
            np.median(sf, axis=1),
            np.percentile(sf, 25, axis=1),
            np.percentile(sf, 75, axis=1),
            ])

    return np.array(features).flatten()


def get_hdf5_file_segments(hdf5_file, artf_file, window_size_sec=10):
    icp_data, icp_sr, icp_start_time_s, icp_end_time_s = read_hdf5(hdf5_file, signal="icp")
    abp_data, abp_sr, abp_start_time_s, abp_end_time_s = read_hdf5(hdf5_file, signal=ABP_KEY)

    global_artefacts, icp_artefacts, abp_artefacts, metadata = read_artf(artf_file, abp_name=ABP_KEY)
    icp_artefacts = sorted(icp_artefacts + global_artefacts, key=lambda x: x.start_time.timestamp())
    abp_artefacts = sorted(abp_artefacts + global_artefacts, key=lambda x: x.start_time.timestamp())

    icp_artefacts_idx = []
    for artefact in icp_artefacts:
        artefact_idx_start, artefact_idx_end = artefact.to_index(icp_sr, window_size_sec, icp_start_time_s)
        icp_artefacts_idx.append(artefact_idx_start)
    icp_artefacts_idx = set(icp_artefacts_idx)

    abp_artefacts_idx = []
    for artefact in abp_artefacts:
        artefact_idx_start, artefact_idx_end = artefact.to_index(abp_sr, window_size_sec, abp_start_time_s)
        abp_artefacts_idx.append(artefact_idx_start)
    abp_artefacts_idx = set(abp_artefacts_idx)

    X = []
    y = []
    original = []

    normal_count = 0
    for i in range(len(icp_data) // int(icp_sr * window_size_sec)):
        segment_start = i * int(icp_sr * window_size_sec)
        segment_end = (i + 1) * int(icp_sr * window_size_sec)
        icp_segment = icp_data[segment_start:segment_end]
        
        is_artefact = segment_start in icp_artefacts_idx
        if not is_artefact and normal_count > len(icp_artefacts_idx) * 1.5:
            continue
        try:
            slow_features = process_segment(icp_segment, icp_sr, is_icp=True)
        except ValueError as e:
            print("ValueError", e)
            continue
        if not is_artefact:
            normal_count += 1

        X.append(slow_features)
        y.append(is_artefact)
        original.append(icp_segment)

    #normal_count = 0
    #for i in range(len(abp_data) // int(abp_sr * window_size_sec)):
    #    segment_start = i * int(abp_sr * window_size_sec)
    #    segment_end = (i + 1) * int(abp_sr * window_size_sec)
    #    abp_segment = abp_data[segment_start:segment_end]
    #    
    #    is_artefact = segment_start in abp_artefacts_idx
    #    if not is_artefact and normal_count > len(abp_artefacts_idx) * 1.5:
    #        continue
    #    try:
    #        slow_features = process_segment(abp_segment, abp_sr, is_icp=False)
    #    except ValueError as e:
    #        print("ValueError", e)
    #        continue
    #    if not is_artefact:
    #        normal_count += 1

    #    X.append(slow_features)
    #    y.append(is_artefact)
    #    original.append(abp_segment)


    return np.array(X), np.array(y), original

# get all hdf5 files in data/2024-03-04
hdf5_files = []
for root, dirs, files in os.walk("data/2024-03-04"):
    for file in files:
        if file.endswith(".hdf5"):
            hdf5_files.append(os.path.join(root, file))
#hdf5_files = hdf5_files[:1]
#hdf5_files = ['data/2024-03-04/TBI_003.hdf5']
# remove TBI_003
hdf5_files = [file for file in hdf5_files if "TBI_003" not in file]

# get all artf files in data/2024-03-04
artf_files = []
for hdf5_file in hdf5_files:
    if not os.path.exists(hdf5_file.replace(".hdf5", ".artf")):
        print(f"ARTF file for {hdf5_file} does not exist.")
        continue
    artf_files.append(hdf5_file.replace(".hdf5", ".artf"))

X_slow = []
y = []
original = []
for i, hdf5_file in enumerate(hdf5_files):
    artf_file = artf_files[i]

    print(f"Processing {hdf5_file}")
    _X, _y, _original = get_hdf5_file_segments(hdf5_file, artf_file)
    X_slow.extend(_X)
    y.extend(_y)
    original.extend(_original)

    print(_X.shape, _y.shape)

# balance classes
artefacts = np.sum(y)
normal = len(y) - artefacts
print("Artefacts", artefacts, "Normal", normal)
X_slow = np.array(X_slow)
y = np.array(y)
artefacts_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]
normal_idx = np.random.choice(normal_idx, artefacts)
idx = np.concatenate([artefacts_idx, normal_idx])
X_slow = X_slow[idx]
y = y[idx]

print("Processing done")
X_slow = np.array(X_slow)
y = np.array(y)
print(X_slow.shape, y.shape)
print("Artefacts", np.sum(y))

X = []
# extract features
for slow_features in X_slow.copy():
    #X.append(extract_features(slow_features))
    features = extract_features(slow_features)
    #features = slow_features.flatten()
    X.append(features)

X = np.array(X)
print(X.shape, y.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


#clf = KNeighborsClassifier(n_neighbors=7)
#clf = SVC()
clf = RandomForestClassifier(n_estimators=200)
#clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(X_train, y_train)
#print(clf.score(X_test, y_test))
print(classification_report(y_test, clf.predict(X_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))
tn, fp, fn, tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
print("TN", tn, "FP", fp, "FN", fn, "TP", tp)

# plot fp segments
fp_idxs = np.where((y_test == 0) & (clf.predict(X_test) == 1))
for idx in fp_idxs[0]:
    plt.plot(original[idx])
    plt.title("False positive")
    plt.show()

# plot fn segments
fn_idxs = np.where((y_test == 1) & (clf.predict(X_test) == 0))
for idx in fn_idxs[0]:
    plt.plot(original[idx])
    plt.title("False negative")
    plt.show()

fp_idx = np.where((y_test == 0) & (clf.predict(X_test) == 1))[0]
fn_idx = np.where((y_test == 1) & (clf.predict(X_test) == 0))[0]

plt.rcParams["figure.figsize"] = (10, 6)

plt.subplot(2, 2, 1)
plt.plot(original[fp_idx[1]])
plt.title("FP")
plt.ylabel("ABP (mmHg)")
#plt.ylabel("ICP (mmHg)")
plt.xlabel("Čas [s]")
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / 100)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)


original_upsampled = upsample_signal(original[fp_idx[1]], 100, 10000)
f, t, Zxx = compute_stft(original_upsampled, sr=10000)
# compute SFA
x = np.abs(Zxx).T
number_features = 14
sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
slow_features = sfa.execute(x)
slow_features = slow_features.T
plt.subplot(2, 2, 2)
plt.imshow(slow_features, aspect='auto', origin='lower',
           extent=[0, slow_features.shape[1], 0, slow_features.shape[0]],
           vmin=0, vmax=1)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, slow_features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("FP STFT SFA")


plt.subplot(2, 2, 3)
plt.plot(original[fn_idx[1]])
plt.title("FN")
plt.ylabel("ABP (mmHg)")
#plt.ylabel("ICP (mmHg)")
plt.xlabel("Čas [s]")
xticks = plt.gca().get_xticks()
xticks_labels = [f"{int(xtick / 100)} s" for xtick in xticks]
plt.gca().set_xticklabels(xticks_labels)

original_upsampled = upsample_signal(original[fn_idx[1]], 100, 10000)
f, t, Zxx = compute_stft(original_upsampled, sr=10000)
# compute SFA
x = np.abs(Zxx).T
number_features = 14
sfa = mdp.nodes.SFANode(output_dim=number_features, rank_deficit_method='auto')
slow_features = sfa.execute(x)
slow_features = slow_features.T
plt.subplot(2, 2, 4)
plt.imshow(slow_features, aspect='auto', origin='lower',
           extent=[0, slow_features.shape[1], 0, slow_features.shape[0]],
           vmin=0, vmax=1)
plt.xlabel("Čas [s]")
plt.ylabel("Vlastnost")
xticks = np.arange(0, slow_features.shape[1], 200)
xticks_labels = [int(xtick / 200) for xtick in xticks]
plt.xticks(xticks, xticks_labels)
plt.title("FN STFT SFA")

plt.tight_layout(pad=0.5)
plt.show()
