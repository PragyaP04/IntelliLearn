🤖 INTELLILEARN - PDF Summarizer + Quiz Generator + Chatbot
INTELLILEARN is an AI-powered web application built with Streamlit that allows users to:
✅ Upload PDFs and extract content
✅ Automatically generate summaries
✅ Create MCQ quizzes based on the PDF
✅ Ask questions interactively using an intelligent chatbot

This project integrates Hugging Face Transformers, Google Gemini API, and FAISS for embeddings and semantic search.

🚀 Features
PDF Upload & Text Extraction – Supports any PDF file and extracts clean text.

AI-Powered Summarization – Uses facebook/bart-large-cnn for concise summaries.

Dynamic Quiz Generation – Creates MCQs (with answers) using Gemini API.

PDF-based Chatbot – Ask context-based questions about the uploaded PDF using semantic search + Gemini.

Chunk-based Processing – Handles large PDFs by splitting into token-friendly chunks.

🛠️ Tech Stack
Frontend: Streamlit

AI/ML Models:

Hugging Face (BART for Summarization)

Google Gemini (Quiz & Chatbot)

Vector Store: FAISS (via LangChain)

Libraries: PyPDF2, NLTK, HuggingFace Transformers, dotenv

📂 Project Structure
bash
Copy
Edit
├── app.py               # Main Streamlit app
├── .env                 # Store your API key here
└── requirements.txt     # Required dependencies
🔑 Setup Instructions
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/intellilearn.git
cd intellilearn
2️⃣ Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Add your Gemini API Key
Create a .env file in the project root:

ini
Copy
Edit
GEMINI_API_KEY=your_gemini_api_key_here
5️⃣ Run the application
bash
Copy
Edit
streamlit run app.py
🧪 How It Works
Upload PDF → Extracts and cleans text.

Summarization → Splits into chunks & generates summaries using BART.

Quiz Generation → Gemini API creates 5 MCQs (with answers) per request.

Chatbot → Uses FAISS embeddings + Gemini to answer PDF-related queries.

📦 Dependencies
Include these in requirements.txt:

nginx
Copy
Edit
streamlit
PyPDF2
nltk
transformers
google-generativeai
langchain-community
sentence-transformers
python-dotenv
🎯 Future Enhancements
✅ Export summaries and quizzes as downloadable PDFs

✅ Support multiple file uploads

✅ Improve quiz interactivity with scoring system

