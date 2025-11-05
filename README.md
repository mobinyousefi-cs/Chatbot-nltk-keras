# 🤖 Chatbot Project – Built with NLTK & Keras

A **Retrieval-Based Chatbot** implemented using **Natural Language Processing (NLP)** techniques (via NLTK) and a **Neural Network** (via Keras/TensorFlow).  
This chatbot learns from example intents and responses to interact conversationally and intelligently — like a lightweight version of Siri or Alexa!

---

## 🧠 Overview

This project demonstrates how to build a **goal-oriented chatbot** capable of understanding user input, classifying the intent, and generating the most appropriate response.

It uses a simple **bag-of-words model** with **tokenization, stemming**, and an **LSTM-inspired dense neural network** to predict the intent of the user’s query.  
The system is modular and easy to expand — just add new intents to the JSON dataset and retrain.

---

## 🏧 Features

- 🔹 **Retrieval-Based Chatbot** — fast and predictable  
- 🔹 **Deep Learning Classification** using Keras (TensorFlow backend)  
- 🔹 **Text Preprocessing** with NLTK (tokenization, stemming, bag-of-words)  
- 🔹 **Configurable Dataset** in simple JSON format (`intents.json`)  
- 🔹 **Command-Line Interface (CLI)** for chatting and training  
- 🔹 **Modular src/ layout** with `pyproject.toml`, tests, CI, and clear code separation  
- 🔹 **MIT Licensed**, open-source, and ready for research or production use

---

## 📂 Project Structure

```text
chatbot-nltk-keras/
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .editorconfig
├─ .github/workflows/ci.yml
├─ src/
│  └─ chatbot_nltk_keras/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ preprocessing.py
│     ├─ model.py
│     ├─ train.py
│     ├─ chatbot.py
│     └─ data/
│        └─ intents.json
└─ tests/
   └─ test_preprocessing.py
```

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mobinyousefi-cs/chatbot-nltk-keras.git
cd chatbot-nltk-keras
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -e .
```

---

## 🧉 Dataset

The chatbot is trained on a small **`intents.json`** file located in:

```
src/chatbot_nltk_keras/data/intents.json
```

Each intent includes:
- **tag** – category of intent  
- **patterns** – example user phrases  
- **responses** – possible chatbot replies  

Example snippet:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": [
        "Hello! How can I help you today?",
        "Hey there 👋 What can I do for you?"
      ]
    }
  ]
}
```

You can easily extend this file to make the chatbot smarter — just retrain afterwards.

---

## 🏃️‍♂️ Training the Model

Train the chatbot using:

```bash
chatbot-train
```

This command will:
- Load the intents file  
- Preprocess data (tokenization, stemming, one-hot encoding)  
- Train a Keras neural network  
- Save artifacts (model + metadata) into `/artifacts`

Artifacts generated:
- `artifacts/chatbot_model.h5`
- `artifacts/metadata.pkl`

Or run manually:
```bash
python -m chatbot_nltk_keras.train
```

---

## 💬 Chatting with Your Bot

Once trained, start chatting via the command line:

```bash
chatbot-cli
```

Example session:

```text
────────────────────────────────
🤖 Chatbot
────────────────────────────────
Type 'quit' to exit.

You: hi
Bot: Hello! How can I help you today?

You: what can you do?
Bot: I can chat with you, answer basic questions, and demonstrate a simple chatbot built with NLTK & Keras.

You: thanks
Bot: You're very welcome 😊

You: quit
Bot: Goodbye! It was nice talking to you.
```

---

## 🤯 Testing

To verify preprocessing and dataset integrity:

```bash
pytest
```

---

## 🛠️ Tech Stack

| Component | Description |
|------------|--------------|
| **Python 3.10+** | Core programming language |
| **NLTK** | Natural language preprocessing |
| **Keras / TensorFlow** | Deep learning backend |
| **NumPy** | Numerical computations |
| **Rich** | Beautiful CLI output |
| **Pytest** | Automated testing |

---

## 🧮 Extending the Chatbot

You can make this chatbot more powerful by:

- Adding **more intents and responses** to `intents.json`  
- Incorporating **lemmatization** instead of stemming  
- Using **word embeddings (Word2Vec, GloVe)**  
- Adding **contextual memory** for multi-turn conversations  
- Deploying with **Flask / FastAPI** for a web-based chatbot interface  

---

## 👨‍💻 Author

**Mobin Yousefi**  
🌍 *Master’s Student in Computer Science*  
🔗 [GitHub – mobinyousefi-cs](https://github.com/mobinyousefi-cs)  
🧠 Focused on Artificial Intelligence, Deep Learning, and Optimization Algorithms.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

### ✨ “It’s not just a chatbot – it’s the first step toward creating your own digital assistant.”

