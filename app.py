import streamlit as st
import PyPDF2, re, os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, AutoTokenizer
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# -------------------------
# INITIAL SETUP
# -------------------------
nltk.download('punkt')
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found! Please add GEMINI_API_KEY in .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="INTELLILEARN", layout="wide")
st.title("🤖 INTELLILEARN - PDF Summarizer + Quiz + Chatbot")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # Extract Text
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    text = re.sub(r'\s+', ' ', text)
    st.session_state["pdf_text"] = text

    # -------------------------
    # SUMMARIZATION (UPDATED)
    # -------------------------
    if st.button("Summarize PDF"):
        st.write("⏳ Summarizing...")
        model_name = "facebook/bart-large-cnn"
        summarizer = pipeline("summarization", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Huggingface models have around 1024 token limit; use ~900 for safety
        max_token_limit = 900
        sentences = sent_tokenize(text)
        chunks, chunk, token_count = [], [], 0

        for sentence in sentences:
            num_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
            if token_count + num_tokens > max_token_limit:
                if chunk:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                token_count = num_tokens
            else:
                chunk.append(sentence)
                token_count += num_tokens
        if chunk:
            chunks.append(" ".join(chunk))

        summaries = []
        for i, c in enumerate(chunks):
            if len(c.strip()) < 50:  # Skip very short chunks
                st.warning(f"Skipping chunk {i+1}: Too short to summarize.")
                continue
            try:
                output = summarizer(c, max_length=150, min_length=50, do_sample=False)
                # Ensure output is valid before extracting summary
                if output and isinstance(output, list) and 'summary_text' in output[0]:
                    summaries.append(output[0]['summary_text'])
                else:
                    st.warning(f"Skipping chunk {i+1}: No summary generated.")
            except Exception as e:
                st.warning(f"Skipping chunk {i+1} due to error: {e}")

        if summaries:
            final_summary = " ".join(summaries)
            st.subheader("📑 Summary")
            st.write(final_summary)

            # Save in session state
            st.session_state["chunks"] = chunks
            st.session_state["summary"] = final_summary

            # Create FAISS for semantic search
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            docs = [Document(page_content=c) for c in chunks]
            st.session_state["faiss_store"] = FAISS.from_documents(docs, embeddings)
        else:
            st.error("⚠️ No valid summary generated.")

    # -------------------------
    # QUIZ GENERATION (PROGRESSIVE)
    # -------------------------
    if "pdf_text" in st.session_state:
        st.subheader("📝 Generate Quiz from PDF")

        if "quiz_sets" not in st.session_state:
            st.session_state.quiz_sets = []

        gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        if st.button("Generate First 5 Questions"):
            with st.spinner("Generating first 5 quiz questions..."):
                prompt = f"Generate 5 MCQs (4 options each, include correct answers) from this PDF:\n\n{st.session_state['pdf_text']}"
                quiz_block = gemini.generate_content(prompt).text
            st.session_state.quiz_sets.append(quiz_block)
            st.success("✅ First 5 quiz questions generated!")

        if st.session_state.quiz_sets:
            st.write("### 📝 Quiz Questions")
            for i, qset in enumerate(st.session_state.quiz_sets, 1):
                st.markdown(f"**Set {i}:**\n{qset}")

            if st.button("Generate 5 More Questions"):
                with st.spinner("Generating 5 more questions..."):
                    prompt = (
                        "Generate 5 more unique MCQs (4 options each, include correct answers) from this PDF. Avoid repeats:\n\n"
                        f"{st.session_state['pdf_text']}"
                    )
                    more_quiz = gemini.generate_content(prompt).text
                st.session_state.quiz_sets.append(more_quiz)
                st.success("✅ Added 5 more questions!")

    # -------------------------
    # CHATBOT
    # -------------------------
    if "faiss_store" in st.session_state:
        st.subheader("💬 Ask Questions About the PDF")

        suggested = [
            "Which libraries are imported in this project?",
            "Who created this project?",
            "How many rows and columns are there in the dataset?",
            "What is the memory usage of the dataset?",
            "What does the heatmap represent?"
        ]
        st.caption("💡 Suggested questions:")
        cols = st.columns(len(suggested))
        for i, q in enumerate(suggested):
            if cols[i].button(q):
                st.session_state["selected_question"] = q

        query = st.text_input("Type your question here:", value=st.session_state.get("selected_question", ""))

        if query:
            gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")
            results = st.session_state["faiss_store"].similarity_search(query, k=3)
            context = "\n".join([r.page_content for r in results])

            response = gemini.generate_content(f"Context:\n{context}\n\nAnswer clearly: {query}")
            st.write("📚 **Answer:**", response.text)
    else:
        st.warning("⚠️ Please summarize the PDF first before asking questions.")
