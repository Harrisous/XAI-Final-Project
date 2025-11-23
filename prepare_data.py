# code last modified using gemini 3 pro 11/22/2025 18:19
import os
import warnings
import sys
import torch
import numpy as np
import pandas as pd
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from transformers import logging as hf_logging
from datasets import load_dataset
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm

# configuration
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
hf_logging.set_verbosity_error()

MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
DATASET_ID = "renumics/emodb"
TARGET_SAMPLING_RATE = 16000 

# Original data (1) + variants (2) = 3x density (about 1600 points)
AUGMENTATION_FACTOR = 2 

def add_noise(audio_array, noise_level=0.005):
    """Add slight white noise"""
    noise = np.random.randn(len(audio_array))
    augmented = audio_array + noise_level * noise
    return augmented.astype(np.float32)

def shift_pitch(audio_array, sr, steps=0.5):
    """Slightly shift pitch (slower, optional)"""
    # For speed, we mainly use Noise here. If you have strong computing power, you can uncomment the following line
    # return librosa.effects.pitch_shift(audio_array, sr=sr, n_steps=steps)
    return audio_array

def main():
    print(f"🔄 Loading Emotion Model: {MODEL_ID} ...")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).to(torch.device("cpu"))
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    print(f"⬇️ Downloading Dataset: {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"✅ Original size: {len(dataset)} samples.")
    print(f"🚀 Applying Data Augmentation (x{AUGMENTATION_FACTOR+1} density)...")

    embeddings = []
    labels = [] 
    
    # EmoDB label mapping
    label_mapping = {
        'w': 'anger', 'aerger': 'anger', 'anger': 'anger',
        'l': 'boredom', 'langeweile': 'boredom', 'boredom': 'boredom',
        'e': 'disgust', 'ekel': 'disgust', 'disgust': 'disgust',
        'a': 'fear', 'angst': 'fear', 'fear': 'fear',
        'f': 'happiness', 'freude': 'happiness', 'happiness': 'happiness',
        't': 'sadness', 'trauer': 'sadness', 'sadness': 'sadness',
        'n': 'neutral', 'neutral': 'neutral'
    }

    model.eval()

    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            
            # 1. Basic data processing
            original_audio = sample['audio']['array']
            orig_sr = sample['audio']['sampling_rate']
            
            # Label processing
            raw_label = str(sample.get('emotion')).lower()
            label_text = label_mapping.get(raw_label, raw_label)
            
            if orig_sr != TARGET_SAMPLING_RATE:
                original_audio = librosa.resample(original_audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLING_RATE)

            # --- 2. Prepare variants list ---
            # Variant 0: Original audio
            audio_versions = [original_audio]
            
            # Variant 1~N: Noisy audio
            for _ in range(AUGMENTATION_FACTOR):
                # Randomly generate different noise levels (0.001 ~ 0.01)
                random_noise = np.random.uniform(0.001, 0.01) 
                noisy_audio = add_noise(original_audio, noise_level=random_noise)
                audio_versions.append(noisy_audio)

            # --- 3. Batch feature extraction ---
            for audio in audio_versions:
                inputs = feature_extractor(
                    audio, 
                    sampling_rate=TARGET_SAMPLING_RATE, 
                    return_tensors="pt", 
                    padding=True
                )
                
                outputs = model(inputs.input_values)
                hidden_states = outputs.last_hidden_state
                embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
                
                embeddings.append(embedding)
                labels.append(label_text)

    print(f"✅ Extraction complete! Total points: {len(embeddings)}")
    print("Running PCA...")

    # --- PCA ---
    X = np.array(embeddings)
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    # --- Save Assets ---
    joblib.dump(pca, 'pca_model.pkl')
    
    df = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
    df['label'] = labels 
    df.to_csv('emotion_map.csv', index=False)

    print("\n🎉 Success! Generated DENSE emotion_map.csv")

if __name__ == "__main__":
    main()