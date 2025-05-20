# Sentiment-Across-Signals: Neural Networks vs. LLMs

EmotionNet is a class project for *Advanced Topics in Predictive Analytics* at Católica Lisbon. It explores emotion detection from both text and speech using neural networks and compares performance with state-of-the-art large language models (LLMs) such as GPT and Claude, using Python, TensorFlow, and LangChain.

---

##  Project Components

### Part 1: Text-Based Sentiment Analysis

Developed a neural network-based classifier that:
- Performs **binary sentiment classification** (positive/negative)
- Detects **multi-class emotions** (joy, sadness, anger, fear, surprise, etc.)

**Tools & Methods:**
- Framework: TensorFlow (Keras)
- Architectures: Compared CNN, LSTM, and dense models
- Dataset: Used IMDb and GoEmotions datasets
- Evaluation: Accuracy

**Relevant Files:**
- `notebook1.ipynb` – Exploratory notebook showing tested models and justifications
- `notebook_final.ipynb` – Final training pipeline for selected best models
- `best_model_binary.h5` – Trained binary sentiment model
- `best_model_multi.h5` – Trained emotion classification model

---

### Part 2: Speech-to-Text Integration

Extended the system to handle **spoken input** using transcription.

**Approach:**
- Used OpenAI's **Whisper** model (local inference) to transcribe audio to text
- Sent the transcribed text into the sentiment and emotion classifier
- Evaluated how transcription quality impacts performance

**Relevant Files:**
- `whisper.ipynb` – Handles speech-to-text transcription with Whisper
- `Part2_final.ipynb` – Integrates transcriptions into the full sentiment pipeline

---

### Part 3: LLM-Based Sentiment Analysis

Implemented commercial **LLM-based classification** using Claude (Anthropic API).

**Objectives:**
- Built sentiment and emotion prompts using **LangChain**
- Compared outputs from Claude with the neural network model predictions
- Analyzed accuracy, consistency, and cost-efficiency

**Setup:**
- Uses LangChain for structured prompt delivery and response parsing
- Sentiment and emotion are inferred from zero-shot prompts

**Relevant Files:**
- New notebook (to be created, e.g., `llm_comparison.ipynb`) for Claude vs NN comparisons
- Claude API setup and prompts documented inside notebook
- Metrics comparison against model predictions

---

## 🧪 Utility Files

- `model.py` – Contains two functions:
  - `create_ensemble_model()` – Combines binary and multi-class models
  - `predict_ensemble_model()` – Performs joint sentiment + emotion predictions
- `df_merged.csv` – Preprocessed dataset with labeled text entries (used for analysis)

---

## 🛠️ Setup Instructions

1. **Install dependencies**
   ```bash
   pip install tensorflow pandas numpy langchain anthropic openai matplotlib
