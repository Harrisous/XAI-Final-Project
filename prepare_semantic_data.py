# code last modified using gemini 3 pro 11/22/2025 19：44
import warnings
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm

# --- Setup & Config ---
warnings.filterwarnings("ignore")
MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_ID = "dair-ai/emotion" 
SAMPLE_COUNT = 5000 

def main():
    print(f"📚 Loading Semantic Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, output_hidden_states=True)
    
    print(f"⬇️ Downloading Text Dataset: {DATASET_ID}...")

    dataset = load_dataset(DATASET_ID, "split", split="train")
    # randomly select SAMPLE_COUNT samples
    dataset = dataset.shuffle(seed=42).select(range(SAMPLE_COUNT))
    
    embeddings = []
    labels = []
    
    # Dair-ai emotion labels mapping
    label_names = dataset.features['label'].names
    
    print("🚀 Extracting Semantic Embeddings...")
    model.eval()
    
    with torch.no_grad():
        for sample in tqdm(dataset):
            text = sample['text']
            label_id = sample['label']
            label_text = label_names[label_id]
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
            
            # Inference
            outputs = model(**inputs)
            
            # Extract Embeddings (Use the CLS token from the last hidden state to represent the entire sentence semantics)
            # shape: (1, seq_len, hidden_size) -> take [0, 0, :] for CLS
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().numpy()
            
            embeddings.append(cls_embedding)
            labels.append(label_text)
            
    print("✅ Extraction complete! Running PCA...")
    
    # --- PCA for Semantic Space ---
    X = np.array(embeddings)
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)
    
    # --- Save ---
    print("💾 Saving Semantic Assets...")
    joblib.dump(pca, 'semantic_pca.pkl')
    
    df = pd.DataFrame(X_3d, columns=['x', 'y', 'z'])
    df['label'] = labels
    df['text_snippet'] = [t[:50]+"..." for t in dataset['text']] # Save partial text for hover display
    df.to_csv('semantic_map.csv', index=False)
    
    print("🎉 Done! Generated 'semantic_map.csv' and 'semantic_pca.pkl'")

if __name__ == "__main__":
    main()