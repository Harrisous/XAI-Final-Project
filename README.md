# Dual-Stream Emotion XAI

Dual-Stream Emotion XAI is an interactive visualization project that compares two complementary channels of human communication:

- **Acoustic (How you sound)** — prosody, pitch, speed, energy — analyzed with a Wav2Vec2-based emotion recognition model.
- **Semantic (What you say)** — the textual meaning and sentiment — analyzed with a RoBERTa-based sentiment/emotion model plus Whisper ASR for transcription.

This repository contains a Streamlit app that projects deep model embeddings from both streams into two 3D PCA latent spaces and displays them side-by-side using Plotly. The result is an intuitive, explorable interface that helps reveal when models agree or disagree and why.

**Why this matters for Explainable AI (XAI)**

This project contributes to XAI by converting high-dimensional, opaque model representations into human-interpretable visual explanations:

- **Separation of channels**: By keeping acoustic and semantic streams distinct, users can see which stream drives the model's decision (tone vs. meaning).
- **Latent-space visualization**: PCA projections reveal cluster structure and nearest neighbors, allowing users to validate whether similar sounding or similar-meaning inputs are grouped together.
- **Conflict identification**: The UI highlights cross-modal conflicts (e.g., positive wording delivered angrily — sarcasm). This helps practitioners understand model failure modes and dataset biases.
- **Transparent features**: Showing embeddings and the PCA transform makes it easier to audit models for issues like over-reliance on prosody or dataset skew.
- **Reproducible pipeline**: The `prepare_*` scripts and saved PCA models let researchers reproduce the maps, inspect intermediate artifacts, and extend the approach to other languages or datasets.

Features
--------
- Real-time audio input via Streamlit and local ASR (Whisper).
- Acoustic emotion recognition using `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`.
- Semantic sentiment/emotion analysis using `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- Interactive 3D plots for both acoustic and semantic latent spaces (Plotly).

Quick Start
-----------
1. Clone the repository and change directory:

```bash
git clone <repo-url>
cd XAI-Final-Project
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```


3. Install dependencies:

There are two sets of dependencies:

- `requirements_prepare.txt` — packages required to run the data preparation scripts (`prepare_data.py`, `prepare_semantic_data.py`).
- `requirements.txt` — packages required to run the Streamlit application in `app.py`.

Recommended workflow (single virtual environment):

```bash
# create and activate venv (if not already done)
python3 -m venv venv
source venv/bin/activate

# install data-prep deps, generate maps
pip install -r requirements_prepare.txt
python prepare_data.py
python prepare_semantic_data.py

# install app deps (you can install both files at once if you prefer)
pip install -r requirements.txt
```

Notes:
- Installing `requirements_prepare.txt` first keeps the data-prep step lightweight if you plan to run preparation separately (e.g., on a different machine).
- You may install both requirements files into the same environment; ordering is not strict.
- Large transformer models and Whisper may be slow on CPU. For faster performance, run on a machine with a GPU and the matching PyTorch/CUDA build.

4. Prepare the reference data and PCA models (if not already present):

If you haven't generated the reference maps already, run:

```bash
python prepare_data.py
python prepare_semantic_data.py
```

These scripts generate the reference CSV maps and PCA pickles used by the visualizer: `emotion_map.csv`, `semantic_map.csv`, `pca_model.pkl`, `semantic_pca.pkl`.

5. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

Files of Interest
-----------------
- `app.py` — Streamlit application and visualization logic.
- `prepare_data.py` — Builds the acoustic reference map and PCA transformer.
- `prepare_semantic_data.py` — Builds the semantic reference map and PCA transformer.
- `emotion_map.csv`, `semantic_map.csv` — Reference datasets used for plotting.
- `pca_model.pkl`, `semantic_pca.pkl` — Saved PCA models used to project embeddings.
- `requirements.txt`, `requirements_prepare.txt` — Python dependencies for app and preparation scripts.
- `LICENSE` — Project license.

Required Files (what the app expects)
------------------------------------
- `emotion_map.csv`: reference acoustic points. Typical columns: `x`,`y`,`z`,`label`, plus any metadata used during plotting.
- `semantic_map.csv`: reference semantic points. Typical columns: `x`,`y`,`z`,`label`, plus any metadata used during plotting.
- `pca_model.pkl`: saved scikit-learn PCA transformer used to project acoustic embeddings to 3D.
- `semantic_pca.pkl`: saved scikit-learn PCA transformer used to project semantic embeddings to 3D.

These artifacts are produced by `prepare_data.py` and `prepare_semantic_data.py`. If any file is missing the app will prompt you to run the prepare scripts.

Models & Data
-------------
- Acoustic emotion model: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (Hugging Face).
- Semantic sentiment/emotion model: `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face).
- ASR: `openai/whisper-tiny` via Hugging Face `pipeline`.
- Acoustic reference dataset: `renumics/emodb` (Berlin Database of Emotional Speech).
- Semantic reference dataset: `dair-ai/emotion` (Twitter emotion dataset).

Notes & Troubleshooting
-----------------------
- If the app warns that the map files are missing, run the `prepare_data.py` and `prepare_semantic_data.py` scripts to generate the CSVs and PCA pickles.
- Model weights are downloaded on first run by Hugging Face; ensure you have internet access and enough disk space.
- The app processes audio locally; no audio is uploaded by default. Check `app.py` if you plan to change storage or telemetry behavior.

Examples
--------
1. Generate maps and run locally (minimal):

```bash
source venv/bin/activate
pip install -r requirements_prepare.txt
python prepare_data.py
python prepare_semantic_data.py
pip install -r requirements.txt
streamlit run app.py
```

2. Interpreting the UI:
- The **Acoustic Space** shows where your recorded voice lies relative to reference recordings (prosody-driven clusters).
- The **Semantic Space** shows where your transcribed sentence lies relative to labeled text examples (meaning-driven clusters).
- If the two spaces disagree (e.g., your point lands near `joy` in semantic space but `anger` in acoustic space), this flags a cross-modal conflict such as sarcasm or a prosody bias.

License & Credits
-----------------
See the `LICENSE` file for full license details.

Built with Streamlit, Hugging Face Transformers, Plotly, Librosa, scikit-learn, and joblib.

All code generated using Gemini 3 pro.

Maintainer & Credits
--------------------
This project was developed as part of the AIPI590 XAI coursework. Contributions and improvements are welcome via pull requests or issues.
