import streamlit as st
import PyPDF2, re
from transformers import pipeline, AutoTokenizer
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------
# UI SETUP
# -------------------------
st.set_page_config(page_title="INTELLILEARN", layout="wide")
st.title("🤖 INTELLILEARN - PDF Summarizer + Quiz + Chatbot")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # Extract Text from PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    text = re.sub(r'\s+', ' ', text)
    st.session_state["pdf_text"] = text

    # -------------------------
    # SUMMARIZATION
    # -------------------------
    if st.button("Summarize PDF"):
        st.write("⏳ Summarizing...")
        model_name = "facebook/bart-large-cnn"
        summarizer = pipeline("summarization", model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks, overlap = [], 50
        for i in range(0, len(tokens), 1024 - overlap):
            chunk = tokenizer.decode(tokens[i:i+1024], skip_special_tokens=True)
            chunks.append(chunk)

        summaries = [
            summarizer(c, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            for c in chunks[:5]
        ]
        final_summary = " ".join(summaries)
        st.subheader("📑 Summary")
        st.write(final_summary)

        # Store in session
        st.session_state["chunks"] = chunks
        st.session_state["summary"] = final_summary

        # Create FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = [Document(page_content=c) for c in chunks]
        st.session_state["faiss_store"] = FAISS.from_documents(docs, embeddings)

    # -------------------------
    # QUIZ GENERATION FROM PDF (Progressive)
    # -------------------------
    if "pdf_text" in st.session_state:
        st.subheader("📝 Generate Quiz from PDF")

        if "quiz_sets" not in st.session_state:
            st.session_state.quiz_sets = []  # store multiple sets of questions

        api_key_quiz = st.text_input("Enter your Gemini API key for Quiz:", type="password")

        if st.button("Generate First 5 Questions"):
            if api_key_quiz:
                genai.configure(api_key=api_key_quiz)
                gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")
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
                    prompt = f"Generate 5 more unique MCQs (4 options each, include correct answers) from this PDF. Avoid repeats:\n\n{st.session_state['pdf_text']}"
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
            api_key_chat = st.text_input("Enter your Gemini API key for Chatbot:", type="password", key="chat_api")
            if api_key_chat:
                genai.configure(api_key=api_key_chat)
                gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")

                results = st.session_state["faiss_store"].similarity_search(query, k=3)
                context = "\n".join([r.page_content for r in results])

                response = gemini.generate_content(f"Context:\n{context}\n\nAnswer clearly: {query}")
                st.write("📚 **Answer:**", response.text)
            else:
                st.warning("⚠️ Please enter your Gemini API key for chatbot.")
    else:
        st.warning("⚠️ Please summarize the PDF first before asking questions.")
