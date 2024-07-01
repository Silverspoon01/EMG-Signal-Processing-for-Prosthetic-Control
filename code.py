import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Load your dataset
# Replace 'your_dataset.txt' with the actual path or filename of your text file
dataset = pd.read_csv('C:/Users/kavya/OneDrive/Desktop/EMG_data_for_gestures-master/04/2_raw_data_18-03_24.04.16.txt', delimiter='\t')

# Define a bandpass filter function
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, min_padlen=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Calculate a suitable padlen based on the length of the data
    padlen = max(min_padlen, len(data) + min_padlen)  # Ensure padlen is at least the minimum
    
    # Pad the signal
    data_padded = np.pad(data, (padlen, padlen), mode='edge')
    
    # Apply bandpass filter
    y = filtfilt(b, a, data_padded)
    
    return y[padlen:-padlen]


def haar_wavelet_transform(signal):
    coeffs = []
    while len(signal) >= 2:
        if len(signal) % 2 != 0:
            signal = np.append(signal, 0)  # Pad with zero if the length is odd
        avg = (signal[0::2] + signal[1::2]) / 2.0
        diff = (signal[0::2] - signal[1::2]) / 2.0
        coeffs.append(diff)
        signal = avg
    if len(coeffs) == 0:
        return np.array([])
    coeffs.append(signal)  # Approximation coefficients
    return np.concatenate(coeffs)

# Define a moving average function
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Plot the original sEMG signals
fig, axes = plt.subplots(8, 1, figsize=(10, 12))
fig.suptitle('Raw sEMG Signals from 8 Channels')

for i in range(8):
    ax = axes[i]
    ax.plot(dataset.iloc[:, i + 1].values, label=f'Channel {i + 1}', color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Apply bandpass filter and Haar wavelet transform
# Create subplots for filtered signals - First 4 channels
plt.figure(figsize=(10, 8))
plt.suptitle('Filtered sEMG Signals - Channels 1 to 4')

for i in range(4):
    # Apply bandpass filter
    filtered_signal = butter_bandpass_filter(dataset.iloc[:, i + 1].values, lowcut=20, highcut=500, fs=4000)
    
    plt.subplot(4, 1, i + 1)
    plt.plot(filtered_signal, label=f'Channel {i + 1} - Filtered Signal', color='green')
    plt.title(f'Channel {i + 1} - Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Create subplots for filtered signals - Last 4 channels
plt.figure(figsize=(10, 8))
plt.suptitle('Filtered sEMG Signals - Channels 5 to 8')

for i in range(4, 8):
    # Apply bandpass filter
    filtered_signal = butter_bandpass_filter(dataset.iloc[:, i + 1].values, lowcut=20, highcut=500, fs=4000)
    
    plt.subplot(4, 1, i - 3)
    plt.plot(filtered_signal, label=f'Channel {i + 1} - Filtered Signal', color='green')
    plt.title(f'Channel {i + 1} - Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# haar wavelet tranformed signal
# Create subplots for Haar wavelet transformed signals - First 4 channels
# Haar wavelet tranformed signal
# Create subplots for Haar wavelet transformed signals - First 4 channels
plt.figure(figsize=(10, 8))
plt.suptitle('Haar Wavelet Transformed sEMG Signals from 8 Channels')

for i in range(4):
    # Apply bandpass filter
    filtered_signal = butter_bandpass_filter(dataset.iloc[:, i + 1].values, lowcut=20, highcut=500, fs=4000)
    
    # Apply Haar wavelet transform
    transformed_signal = haar_wavelet_transform(filtered_signal)
    
    plt.subplot(4, 1, i + 1)
    plt.plot(np.arange(len(transformed_signal)), transformed_signal, label=f'Channel {i + 1} - Wavelet Transform', color='red')
    plt.title(f'Channel {i + 1} - Wavelet Transformed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(10, 8))
plt.suptitle('Haar Wavelet Transformed sEMG Signals from Channels 5 to 8')

for i in range(4, 8):
    # Apply bandpass filter
    filtered_signal = butter_bandpass_filter(dataset.iloc[:, i + 1].values, lowcut=20, highcut=500, fs=4000)
    
    # Apply Haar wavelet transform
    transformed_signal = haar_wavelet_transform(filtered_signal)
    
    plt.subplot(4, 1, i - 3)
    plt.plot(np.arange(len(transformed_signal)), transformed_signal, label=f'Channel {i + 1} - Wavelet Transform', color='red')
    plt.title(f'Channel {i + 1} - Wavelet Transformed Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Apply moving average to the Haar wavelet transformed signal - First 4 channels
plt.figure(figsize=(10, 8))
plt.suptitle('sEMG Signals with Moving Average (First 4 Channels)')

window_size = 10  # You can adjust the window size as needed

for i in range(4):
    
    
    # Apply moving average to the transformed signal
    smoothed_signal = moving_average(transformed_signal, window_size)
    
    plt.subplot(4, 1, i + 1)
    plt.plot(np.arange(len(smoothed_signal)), smoothed_signal, label=f'Channel {i + 1} - Wavelet Transform + Moving Average', color='purple')
    plt.title(f'Channel {i + 1} - Signal with Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Apply moving average to the Haar wavelet transformed signal - Next 4 channels
plt.figure(figsize=(10, 8))
plt.suptitle('sEMG Signals with Moving Average (Channels 5 to 8)')

for i in range(4, 8):
    
    # Apply moving average to the transformed signal
    smoothed_signal = moving_average(transformed_signal, window_size)
    
    plt.subplot(4, 1, i - 3)
    plt.plot(np.arange(len(smoothed_signal)), smoothed_signal, label=f'Channel {i + 1} - Wavelet Transform + Moving Average', color='purple')
    plt.title(f'Channel {i + 1} - Signal with Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Feature extraction using moving average
processed_features = dataset.iloc[:, 1:9].apply(lambda x: moving_average(butter_bandpass_filter(x.values, lowcut=20, highcut=500, fs=4000), window_size), axis=1)

# Remove rows with empty arrays (signals that caused issues)
processed_features = processed_features[processed_features.apply(len) > 0]

# Combine processed features
X = np.vstack(processed_features.values)
y = dataset.loc[processed_features.index, 'class'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a K-Nearest Neighbors (KNN) classifier
clf = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)