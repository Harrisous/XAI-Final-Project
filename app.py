# code last modified using gemini 3 pro 11/22/2025 22:19
import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import librosa
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForAudioClassification, pipeline

# --- 0. Suppress Warnings ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# --- Page Config ---
st.set_page_config(page_title="Dual-Stream Emotion XAI", layout="wide")

# --- Constants ---
ACOUSTIC_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SEMANTIC_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
TARGET_SAMPLING_RATE = 16000

# --- Load Resources ---
@st.cache_resource
def load_models():
    print("🔄 Loading All AI Models...")
    
    # 1. Acoustic (Audio)
    feat_ext = AutoFeatureExtractor.from_pretrained(ACOUSTIC_MODEL_ID)
    ac_model = AutoModelForAudioClassification.from_pretrained(ACOUSTIC_MODEL_ID, output_hidden_states=True)
    
    # 2. ASR (Whisper)
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    # 3. Semantic (Text) - Manually loaded to get embeddings
    tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_MODEL_ID)
    # output_hidden_states=True
    sem_model = AutoModelForSequenceClassification.from_pretrained(SEMANTIC_MODEL_ID, output_hidden_states=True)
    
    return feat_ext, ac_model, asr_pipe, tokenizer, sem_model

@st.cache_data
def load_maps():
    data = {}
    
    # EmoDB / Wav2Vec2 id to string
    id_to_emotion = {
        0: "anger", 
        1: "boredom", 
        2: "disgust", 
        3: "fear", 
        4: "happiness", 
        5: "neutral", 
        6: "sadness",
        "0": "anger", "1": "boredom", "2": "disgust", "3": "fear", 
        "4": "happiness", "5": "neutral", "6": "sadness"
    }

    def _fix_labels(df, col_name):
        """id to label conversion if needed"""
        if col_name not in df.columns:
            return df
        df[col_name] = df[col_name].astype(str)
        
        if df[col_name].iloc[0].isdigit():
             df[col_name] = df[col_name].map(lambda x: id_to_emotion.get(x, x))
             
        return df

    try:
        # --- 1. Load Acoustic Assets ---
        df_ac = pd.read_csv("emotion_map.csv")
        # ✅ FIX
        df_ac = _fix_labels(df_ac, "label")
        data['ac_df'] = df_ac
        data['ac_pca'] = joblib.load("pca_model.pkl")
        
        # --- 2. Load Semantic Assets ---
        df_sem = pd.read_csv("semantic_map.csv")
        # ✅ FIX
        df_sem = _fix_labels(df_sem, "label")
        data['sem_df'] = df_sem
        data['sem_pca'] = joblib.load("semantic_pca.pkl")
        
        return data
    except FileNotFoundError:
        return None

# Initialize
feat_ext, ac_model, asr_pipe, tokenizer, sem_model = load_models()
maps = load_maps()

# --- Helper: 3D Plotting Function ---
def plot_3d_space(df, user_coords, title, label_col, user_label, color_map=None):
    """Generic function to plot 3D spaces"""
    fig = px.scatter_3d( 
        df, x="x", y="y", z="z", color=label_col, 
        title=title, template="plotly_white", opacity=0.4,
        color_discrete_map=color_map, height=600,
        hover_data=[label_col]
    )
    fig.update_traces(marker=dict(size=4))

    if user_coords is not None:
        fig.add_trace(
            go.Scatter3d( 
                x=[user_coords[0][0]], y=[user_coords[0][1]], z=[user_coords[0][2]], 
                mode='markers+text',
                marker=dict(size=18, color='black', symbol='diamond', line=dict(width=2, color='white')), 
                name='YOU', text=[user_label], textposition="top center",
                textfont=dict(color='black', size=14, family="Arial Black")
            )
        )
    
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)), margin=dict(l=0, r=0, b=0, t=30))
    return fig

# --- Main UI ---
st.title("🧠 Dual-Stream XAI: Sound vs. Meaning")
st.markdown("Comparing **how you sound** (Acoustic) against **what you say** (Semantic) in two distinct latent spaces.")

# Input Section
col_input, col_viz = st.columns([1, 2.5])

with col_input:
    st.subheader("🎤 Input")
    audio_value = st.audio_input("Record Voice")
    
    # State holders
    ac_prob_display = None
    sem_prob_display = None
    transcription = ""
    
    # Coordinates holders
    ac_coords = None
    sem_coords = None
    
    # Labels
    ac_label = None
    sem_label_sentiment = None # RoBERTa gives sentiment (pos/neg)

    if audio_value:
        with st.spinner("Analyzing Dual Streams..."):
            try:
                # --- Stream 1: Acoustic ---
                y, sr = librosa.load(audio_value, sr=TARGET_SAMPLING_RATE)
                ac_inputs = feat_ext(y, sampling_rate=TARGET_SAMPLING_RATE, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    ac_out = ac_model(ac_inputs.input_values)
                    # Label
                    ac_id = torch.argmax(ac_out.logits, dim=-1).item()
                    ac_label = ac_model.config.id2label[ac_id]
                    # Embedding & PCA
                    ac_emb = torch.mean(ac_out.hidden_states[-1], dim=1).squeeze().numpy()
                    if maps: ac_coords = maps['ac_pca'].transform([ac_emb])

                # --- Stream 2: Semantic ---
                # A. Transcribe
                asr_out = asr_pipe(y, generate_kwargs={"language": "english"})
                transcription = asr_out["text"]
                
                # B. Semantic Embedding
                sem_inputs = tokenizer(transcription, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    sem_out = sem_model(**sem_inputs)
                    # Label (Sentiment)
                    sem_id = torch.argmax(sem_out.logits, dim=-1).item()
                    sem_label_sentiment = sem_model.config.id2label[sem_id] # negative, neutral, positive
                    # Embedding (CLS token) & PCA
                    sem_emb = sem_out.hidden_states[-1][:, 0, :].squeeze().numpy()
                    if maps: sem_coords = maps['sem_pca'].transform([sem_emb])
            
            except Exception as e:
                st.error(f"Error: {e}")

    # Display Results Panel
    if ac_label and sem_label_sentiment:
        st.divider()
        st.write(f"**🗣️ Transcript:**")
        st.caption(f'"{transcription}"')
        
        st.write("---")
        st.write(f"**🔊 Tone:**")
        st.markdown(f"### <span style='color:#1E90FF'>{ac_label.upper()}</span>", unsafe_allow_html=True)
        
        st.write(f"**📝 Sentiment:**")
        sentiment_color = "green" if "positive" in sem_label_sentiment else "red" if "negative" in sem_label_sentiment else "gray"
        st.markdown(f"### <span style='color:{sentiment_color}'>{sem_label_sentiment.upper()}</span>", unsafe_allow_html=True)

with col_viz:
    if maps is None:
        st.warning("Please run 'prepare_data.py' AND 'prepare_semantic_data.py' first.")
    else:
        # Tabs for Acoustic and Semantic
        tab1, tab2 = st.tabs(["🔊 Acoustic Space (The Voice)", "📝 Semantic Space (The Words)"])
        
        with tab1:
            # Acoustic Plot
            ac_colors = {"anger": "#FF4B4B", "happiness": "#FFD700", "happy": "#FFD700", "sadness": "#1E90FF", "sad": "#1E90FF", "neutral": "#E0E0E0", "fear": "#9370DB", "disgust": "#228B22"}
            
            fig_ac = plot_3d_space(
                maps['ac_df'], 
                ac_coords, 
                "Acoustic Latent Space (Wav2Vec2)", 
                "label", 
                f"YOU ({ac_label})",
                ac_colors
            )
            st.plotly_chart(fig_ac, width='stretch')
            st.info("This map groups sounds by **prosody** (pitch, speed, energy). 'Happy' sounds cluster together regardless of words.")

        with tab2:
            # Semantic Plot
            # Dair-ai labels: sadness, joy, love, anger, fear, surprise
            sem_colors = {"joy": "#FFD700", "love": "#FF69B4", "anger": "#FF4B4B", "sadness": "#1E90FF", "fear": "#9370DB", "surprise": "#FFA500"}
            
            fig_sem = plot_3d_space(
                maps['sem_df'], 
                sem_coords, 
                "Semantic Latent Space (RoBERTa)", 
                "label", 
                f"YOU ({sem_label_sentiment})",
                sem_colors
            )
            st.plotly_chart(fig_sem, width='stretch')
            st.info("This map groups sentences by **meaning**. Even if you say it angrily, a 'happy sentence' will land in the Joy/Love cluster.")

# --- Footer: Project Documentation ---
st.divider()

st.header("📘 Project Technical Documentation")

with st.expander("1. Project Significance & XAI Goal", expanded=True):
    st.markdown("""
    ### Why this matters?
    Human communication is **multimodal**. When we speak, we convey information through two distinct channels:
    1.  **Acoustic (How we say it):** Pitch, speed, volume, and timbre (Prosody).
    2.  **Semantic (What we say):** The actual vocabulary and grammar (Linguistics).
    
    Standard AI models often treat these separately. This project demonstrates **Dual-Stream Explainable AI (XAI)**. 
    By visualizing these two latent spaces side-by-side, we reveal how AI resolves conflicts (e.g., Sarcasm: *Positive words* + *Negative tone*) and demonstrate the **Cross-Lingual universality** of emotional acoustic features.
    """)

with st.expander("2. Datasets & Data Augmentation"):
    st.markdown("""
    ### Acoustic Data (The Sound) 
    * **Source:** `renumics/emodb` https://huggingface.co/datasets/renumics/emodb (Berlin Database of Emotional Speech)(Only opensource dataset for emotional speech in German).
    * **Nature:** 535 acted emotional sentences in German.
    * **Why German?** To prove **Universality**. The fact that an English speaker's anger maps correctly to German data proves that acoustic emotion features are independent of language.
    * **Data Augmentation:** To visualize robust clusters, we applied **Gaussian Noise Injection** (White Noise) to generate 3x density (approx. 1600 points), simulating real-world noisy environments.

    ### Semantic Data (The Meaning)
    * **Source:** `dair-ai/emotion` https://huggingface.co/datasets/dair-ai/emotion.
    * **Nature:** 16,000+ English Twitter messages labeled with emotions (Joy, Sadness, Anger, Fear, Love, Surprise).
    * **Usage:** Used to build the semantic reference map for the text analysis stream.
    """)

with st.expander("3. Model Architecture"):
    st.markdown("""
    We utilize a **Dual-Stream Pipeline** with Late Fusion:
    
    | Component | Model ID | Role |
    | :--- | :--- | :--- |
    | **The Ear (Acoustic)** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | A fine-tuned **Wav2Vec 2.0** model that extracts deep acoustic embeddings (1024-dim) from raw waveforms. |
    | **The Brain (Semantic)** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | A **RoBERTa** Transformer trained on millions of tweets to understand text sentiment and nuance. (768-dim)|
    """)

with st.expander("4. Visualization Principle (PCA)"):
    st.markdown("""
    ### From 1024/768 Dimensions to 3D
    Deep Learning models "think" in high-dimensional vectors (Embeddings). For example, Wav2Vec2 represents a second of audio as a vector of **1024 numbers**, and RoBERTa represents text as a vector of **768 numbers**. Humans cannot see these high dimensions.
    
    We use **Principal Component Analysis (PCA)** to make this interpretable:
    1.  **Dimensionality Reduction:** PCA finds the 3 "directions" (Principal Components) where the data varies the most. Variation means information. 
    2.  **Projection:** We compress the 1024/768-dimensional math down to X, Y, and Z coordinates.
    3.  **Result:** Similar emotions cluster together. If your voice dot lands in the "Red Cluster", it means your vector is mathematically similar to the vectors of angry people.
    """)

st.markdown("---")
st.caption("© 2025 AIPI590 XAI Project | Built with Streamlit, Hugging Face, & Plotly")