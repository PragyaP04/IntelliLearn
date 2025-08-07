# 🤖 INTELLILEARN - AI PDF Summarizer + Quiz Generator + Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

**INTELLILEARN** is an AI-powered web app that allows users to upload any PDF, get a smart summary, generate multiple-choice quiz questions, and ask questions about the content — all in one seamless interface.

---

## 🎬 Demo

[![Watch the demo](http://img.youtube.com/vi/S92SnSoVWvA/0.jpg)](https://youtu.be/S92SnSoVWvA)

---

## 🔥 Features

- 📄 Upload any PDF and extract content
- 📑 Get AI-generated summaries using `facebook/bart-large-cnn`
- 📝 Generate MCQs using Gemini 1.5 Flash (Google Generative AI)
- 💬 Ask questions via a built-in chatbot using FAISS + Gemini
- 📚 Smart search with semantic understanding of PDF content
- ⚙️ Easy-to-use UI powered by Streamlit

---

## 🛠️ Tech Stack

| Component       | Technology                  |
|----------------|-----------------------------|
| Frontend       | Streamlit                   |
| Backend        | Python                      |
| Summarization  | HuggingFace Transformers (BART) |
| Quiz & Chatbot | Gemini 1.5 Flash (Google Generative AI) |
| Embedding      | HuggingFace MiniLM Embeddings |
| Search Engine  | FAISS                       |
| NLP Toolkit    | NLTK                        |
| PDF Handling   | PyPDF2                      |

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/intellilearn.git
cd intellilearn
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Setup .env
Create a .env file in the root directory with your Gemini API key:

ini
Copy
Edit
GEMINI_API_KEY=your_gemini_api_key_here
4. Run the App
bash
Copy
Edit
streamlit run app.py


