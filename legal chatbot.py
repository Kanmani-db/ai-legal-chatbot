import streamlit as st
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Legal Chatbot", page_icon="⚖️", layout="centered")

st.title("⚖ AI Legal Knowledge Assistant")
st.caption("Indian Constitutional & Legal Awareness Platform")

st.warning("⚠ This chatbot provides general legal information only. Not professional legal advice.")

# ------------------------------------------------
# LEGAL DIAGRAM FUNCTION
# ------------------------------------------------
def show_legal_diagram(question):
    q = question.lower()

    if "article 21" in q:
        st.markdown("### 📊 Article 21 Explained")
        st.markdown("""
        **Right to Life**
        ↓  
        Includes:
        - Right to live with dignity  
        - Right to privacy  
        - Right to livelihood  
        """)

    elif "fundamental rights" in q:
        st.markdown("### 📊 Fundamental Rights")
        st.markdown("""
        6 Types:
        - Right to Equality  
        - Right to Freedom  
        - Right against Exploitation  
        - Right to Religion  
        - Cultural & Educational Rights  
        - Right to Constitutional Remedies  
        """)

    elif "habeas corpus" in q:
        st.markdown("### 📊 Habeas Corpus")
        st.markdown("""
        Illegal Detention  
        ↓  
        Court Order  
        ↓  
        Person Produced Before Judge  
        ↓  
        Release if unlawful  
        """)

# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
st.sidebar.title("⚙ Settings")

languages = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn"
}

selected_language = st.sidebar.selectbox("🌍 Select Language", list(languages.keys()))
lang_code = languages[selected_language]

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []

# ------------------------------------------------
# LEGAL CATEGORIES
# ------------------------------------------------
st.markdown("### 📚 Legal Categories")

legal_examples = {
    "Criminal Law": [
        "What is FIR?",
        "What is bail?",
        "What are my rights if arrested?"
    ],
    "Cyber Law": [
        "How to file cybercrime complaint?",
        "What is online fraud?",
        "What to do if account is hacked?"
    ],
    "Consumer Law": [
        "How to file complaint in consumer court?",
        "What are consumer rights?",
        "What if product is defective?"
    ],
    "Women Protection": [
        "What is domestic violence law?",
        "What is POSH Act?",
        "How to report harassment?"
    ],
    "Constitutional Rights": [
        "What are fundamental rights?",
        "What is Article 21?",
        "What is Habeas Corpus?"
    ]
}

for category, questions in legal_examples.items():
    with st.expander(category):
        for q in questions:
            if st.button(q, key=q):
                st.session_state["auto_question"] = q

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# ------------------------------------------------
# LOAD KNOWLEDGE BASE
# ------------------------------------------------
@st.cache_resource
def load_knowledge():
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        docs = f.readlines()

    embeddings = embedding_model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return docs, index

documents, index = load_knowledge()

# ------------------------------------------------
# CHAT MEMORY
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------
# INPUT SECTION
# ------------------------------------------------
if "auto_question" in st.session_state:
    user_input = st.session_state["auto_question"]
    del st.session_state["auto_question"]
else:
    user_input = st.chat_input("Ask your legal question...")

# ------------------------------------------------
# MAIN CHAT
# ------------------------------------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    question_en = GoogleTranslator(source='auto', target='en').translate(user_input)

    q_embed = embedding_model.encode([question_en])
    _, indices = index.search(np.array(q_embed), k=2)
    context = " ".join([documents[i] for i in indices[0]])

    prompt = f"""
    You are a legal knowledge assistant.
    Explain clearly in simple language.
    Do not provide legal advice.

    Context:
    {context}

    Question:
    {question_en}
    """

    with st.spinner("⚖ Analyzing..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                }
            )

            data = response.json()
            answer_en = data.get("response", "⚠ No valid response from model.")

        except:
            st.error("❌ Ollama not running. Run: ollama serve")
            answer_en = "Server not available."

    # Translate
    if lang_code != "en":
        final_answer = GoogleTranslator(source='auto', target=lang_code).translate(answer_en)
    else:
        final_answer = answer_en

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

    with st.chat_message("assistant"):
        st.markdown(final_answer)
        show_legal_diagram(user_input)

        # 🔊 Voice Output
        try:
            audio_file = f"response_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=final_answer, lang=lang_code)
            tts.save(audio_file)
            st.audio(audio_file)
        except:
            st.warning("🔊 Voice output not available")

# ------------------------------------------------
# CONSTITUTION SECTION
# ------------------------------------------------
st.markdown("---")
st.markdown("## 📚 Constitution & Amendments Knowledge Section")

st.markdown("### 💡 Try These Questions")

constitutional_examples = [
    "What is Article 21 in simple words?",
    "Explain Article 19 with example",
    "What is Habeas Corpus?",
    "What is 42nd Amendment?",
    "What are Fundamental Rights?",
    "Difference between Rights and Duties"
]

cols = st.columns(2)

for i, q in enumerate(constitutional_examples):
    if cols[i % 2].button(q, key="const_" + q):
        st.session_state["constitution_question"] = q

constitution_input = st.text_input("Ask about Constitution:")

if "constitution_question" in st.session_state:
    constitution_input = st.session_state["constitution_question"]
    del st.session_state["constitution_question"]

if constitution_input:

    with st.spinner("📖 Fetching..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": constitution_input,
                    "stream": False
                }
            )

            data = response.json()
            answer = data.get("response", "No response")

        except:
            answer = "❌ Ollama not running"

    st.markdown("### 📜 Answer")
    st.write(answer)
    show_legal_diagram(constitution_input)

    # 🔊 Voice Output
    try:
        audio_file = f"response_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=answer, lang=lang_code)
        tts.save(audio_file)
        st.audio(audio_file)
    except:
        pass