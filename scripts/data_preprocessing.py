import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Feature Extraction Functions ---
def extract_mfcc(y, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

def extract_mel_spectrogram(y, sr, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return np.mean(librosa.power_to_db(mel_spec, ref=np.max).T, axis=0)

def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma.T, axis=0)

def extract_spectral_contrast(y, sr):
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(contrast.T, axis=0)

# Combine Features
def extract_features(audio_path, sr=22050, n_mfcc=40):
    y, _ = librosa.load(audio_path, sr=sr, duration=5.0)
    mfcc = extract_mfcc(y, sr, n_mfcc)
    mel = extract_mel_spectrogram(y, sr)
    chroma = extract_chroma(y, sr)
    spectral_contrast = extract_spectral_contrast(y, sr)
    return np.concatenate([mfcc, mel, chroma, spectral_contrast])

# --- Data Augmentation ---
def augment_audio(y, sr):
    augmented = []
    # Original
    augmented.append(y)
    # Time Stretch
    augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    # Pitch Shift
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
    # Noise Injection
    noise = 0.005 * np.random.randn(len(y))
    augmented.append(y + noise)
    return augmented

# --- Preprocessing Data ---
def preprocess_data(data_dir, labels_path=None, save_features=True, sr=22050, n_mfcc=40, augment=True, normalize=True, reduce_dims=True):
    # Load labels
    if labels_path:
        df = pd.read_csv(labels_path)
    else:
        df = pd.DataFrame({'idx': os.listdir(data_dir)})

    features = []
    labels = []

    for idx in df['idx']:
        file_path = os.path.join(data_dir, f"{idx}.wav")
        
        try:
            y, _ = librosa.load(file_path, sr=sr, duration=5.0)
            # Augmentation
            audio_variants = augment_audio(y, sr) if augment else [y]
            
            for y_variant in audio_variants:
                feature = extract_features(file_path, sr, n_mfcc)
                features.append(feature)
                
                if labels_path:
                    labels.append(df.loc[df['idx'] == idx, 'class'].values[0])
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    features = np.array(features)
    
    # Normalize features
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    # Dimensionality Reduction
    if reduce_dims:
        pca = PCA(n_components=20)  # Reduce to 20 dimensions
        features = pca.fit_transform(features)

    # Encode labels
    if labels_path:
        labels = np.array([1 if label == "RightWhale" else 0 for label in labels])

    # Save features and labels
    if save_features:
        os.makedirs("outputs", exist_ok=True)
        np.save("outputs/features.npy", features)
        if labels_path:
            np.save("outputs/labels.npy", labels)

    return features, labels if labels_path else features

# --- Generate Data for CNN, SVM, RNN ---
def preprocess_cnn_data(data_dir, labels_path=None):
    features, labels = preprocess_data(data_dir, labels_path, augment=False, normalize=True, reduce_dims=False)
    features = features.reshape(-1, 1, features.shape[1])  # Adjust dimensions for CNN
    return features, labels

def preprocess_rnn_data(data_dir, labels_path=None):
    features, labels = preprocess_data(data_dir, labels_path, augment=False, normalize=True, reduce_dims=False)
    features = features.reshape(features.shape[0], -1, 1)  # Adjust dimensions for RNN
    return features, labels

def preprocess_svm_data(data_dir, labels_path=None):
    features, labels = preprocess_data(data_dir, labels_path, augment=False, normalize=True, reduce_dims=True)
    return features, labels