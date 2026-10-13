import streamlit as st
import faiss
import numpy as np
import requests
import edge_tts
import asyncio
import uuid
import os

from sentence_transformers import SentenceTransformer

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LegalAI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b1020;
    color: #e5e7eb;
}

.block-container {
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 60px;
}

section[data-testid="stSidebar"] {
    background: #080d18;
    border-right: 1px solid #1f2937;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #94a3b8;
}

.notice {
    background: #171b27;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 25px;
    color: #cbd5e1;
}

.dashboard-card {
    background: #111827;
    border: 1px solid #263244;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    min-height: 110px;
}

.dashboard-number {
    font-size: 28px;
    font-weight: 700;
    color: #60a5fa;
}

.dashboard-label {
    font-size: 13px;
    color: #94a3b8;
    margin-top: 5px;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    margin-bottom: 12px;
}

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #172554 !important;
    border: 1px solid #1e40af;
}

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #111827 !important;
    border: 1px solid #263244;
}

[data-testid="stChatInput"] {
    border-radius: 15px;
}

.stButton > button {
    background: #111827;
    color: #e5e7eb;
    border: 1px solid #334155;
    border-radius: 10px;
}

.stButton > button:hover {
    background: #172554;
    border-color: #60a5fa;
    color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    padding-top: 40px;
    padding-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚖️ LegalAI")

    st.caption("Indian Legal Knowledge Assistant")

    st.divider()

    st.markdown("### ⚙️ Settings")

    languages = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn"
    }

    selected_language = st.selectbox(
        "🌍 Language",
        list(languages.keys())
    )

    lang_code = languages[selected_language]

    st.divider()

    st.markdown("### 🧠 AI System")

    st.success("FAISS Search")
    st.success("Llama 3")
    st.success("Local Translation")

    st.divider()

    st.markdown("### 🔊 Voice")

    voice_speed = st.slider(
        "Speaking Speed",
        min_value=-10,
        max_value=30,
        value=10,
        step=5
    )

    if voice_speed >= 0:
        voice_rate = f"+{voice_speed}%"
    else:
        voice_rate = f"{voice_speed}%"

    st.caption(f"Voice speed: {voice_rate}")

    st.divider()

    st.markdown("### 📊 Session")

    st.metric(
        "Questions",
        len(st.session_state.messages) // 2
    )

    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "⚠️ For educational and legal awareness purposes only."
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
⚖️ LegalAI
</div>

<div class="hero-subtitle">
Indian Constitutional & Legal Knowledge Assistant
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("""
<div class="notice">
⚠️ <b>Important:</b> This AI provides general legal information
for educational purposes. It is not a substitute for professional
legal advice.
</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

st.markdown("### 📊 Knowledge Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-number">500+</div>
        <div class="dashboard-label">Legal Topics</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-number">100+</div>
        <div class="dashboard-label">Constitutional Topics</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-number">3</div>
        <div class="dashboard-label">Languages</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="dashboard-card">
        <div class="dashboard-number">AI</div>
        <div class="dashboard-label">Llama 3</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
# LEGAL DIAGRAM
# ============================================================

def show_legal_diagram(question):

    q = question.lower()

    if "article 21" in q:

        st.markdown("#### 📊 Article 21")

        st.info("""
**Right to Life and Personal Liberty**

Protection of Life  
↓  
Personal Liberty  
↓  
Life with Dignity  
↓  
Privacy  
↓  
Fair Legal Procedure
""")

    elif "fundamental rights" in q:

        st.markdown("#### 📊 Fundamental Rights")

        st.info("""
Right to Equality  
↓  
Right to Freedom  
↓  
Right against Exploitation  
↓  
Right to Freedom of Religion  
↓  
Cultural & Educational Rights  
↓  
Right to Constitutional Remedies
""")

    elif "habeas corpus" in q:

        st.markdown("#### 📊 Habeas Corpus")

        st.info("""
Illegal Detention  
↓  
Court Intervention  
↓  
Person Produced Before Court  
↓  
Court Examines Detention  
↓  
Release if Detention is Unlawful
""")


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_ollama(prompt):

    response = requests.post(

        "http://localhost:11434/api/generate",

        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },

        # Give the local model more time, especially when running on CPU.
        # requests timeout is (connection timeout, response-read timeout).
        timeout=(10, 300)
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        ""
    ).strip()

    if not answer:
        return "⚠️ The AI returned an empty response."

    return answer


# ============================================================
# LOCAL TRANSLATION USING OLLAMA
# ============================================================

def translate_with_ollama(text, target_language):

    if target_language == "en":
        return text

    language_names = {
        "hi": "Hindi",
        "kn": "Kannada"
    }

    target_name = language_names.get(
        target_language,
        "English"
    )

    translation_prompt = f"""
You are a professional translator.

Translate the following text into {target_name}.

Rules:
- Translate accurately.
- Preserve the original meaning.
- Do not add explanations.
- Do not remove important legal information.
- Keep legal terms accurate.
- Keep bullet points and numbering.
- Return ONLY the translated text.

TEXT:
{text}

TRANSLATION:
"""

    return ask_ollama(
        translation_prompt
    )


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_model()


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_knowledge():

    try:

        with open(
            "knowledge_base.txt",
            "r",
            encoding="utf-8"
        ) as f:

            docs = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

    except FileNotFoundError:

        st.error(
            "❌ knowledge_base.txt was not found."
        )

        return [], None

    if not docs:

        st.error(
            "❌ knowledge_base.txt is empty."
        )

        return [], None

    embedding_file = "embeddings.npy"

    if os.path.exists(embedding_file):

        try:

            embeddings = np.load(
                embedding_file
            )

            if len(embeddings) != len(docs):

                raise ValueError(
                    "Knowledge base changed"
                )

        except Exception:

            embeddings = embedding_model.encode(
                docs,
                batch_size=32,
                show_progress_bar=False
            )

            embeddings = np.array(
                embeddings
            )

            np.save(
                embedding_file,
                embeddings
            )

    else:

        embeddings = embedding_model.encode(
            docs,
            batch_size=32,
            show_progress_bar=False
        )

        embeddings = np.array(
            embeddings
        )

        np.save(
            embedding_file,
            embeddings
        )

    embeddings = embeddings.astype(
        "float32"
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return docs, index


documents, index = load_knowledge()


# ============================================================
# LEGAL CATEGORIES
# ============================================================

st.markdown("### 📚 Explore Legal Topics")

categories = {

    "⚖️ Criminal Law":
        "FIR, bail, arrest and criminal procedures",

    "💻 Cyber Law":
        "Online fraud, hacking and cybercrime",

    "🛒 Consumer Law":
        "Consumer rights, defective products and complaints",

    "👩 Women Protection":
        "Domestic violence, POSH and harassment",

    "📜 Constitutional Law":
        "Fundamental rights, Articles, duties and writs"
}


for category, description in categories.items():

    with st.expander(category):

        st.caption(description)

        if category == "⚖️ Criminal Law":

            questions = [
                "What is FIR?",
                "What is bail?",
                "What are my rights if arrested?"
            ]

        elif category == "💻 Cyber Law":

            questions = [
                "How to file cybercrime complaint?",
                "What is online fraud?",
                "What to do if account is hacked?"
            ]

        elif category == "🛒 Consumer Law":

            questions = [
                "How to file complaint in consumer court?",
                "What are consumer rights?",
                "What if product is defective?"
            ]

        elif category == "👩 Women Protection":

            questions = [
                "What is domestic violence law?",
                "What is POSH Act?",
                "How to report harassment?"
            ]

        else:

            questions = [
                "What are fundamental rights?",
                "What is Article 21?",
                "What is Habeas Corpus?"
            ]

        for q in questions:

            if st.button(
                q,
                key="topic_" + q,
                use_container_width=True
            ):

                st.session_state[
                    "auto_question"
                ] = q

                st.rerun()


# ============================================================
# CHAT HEADER
# ============================================================

st.divider()

st.markdown("### 💬 Ask LegalAI")

st.caption(
    "Ask questions about Indian law, constitutional rights, "
    "cyber law, consumer law and more."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:

    avatar = (
        "👤"
        if msg["role"] == "user"
        else "⚖️"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):

        st.markdown(
            msg["content"]
        )


# ============================================================
# INPUT
# ============================================================

if "auto_question" in st.session_state:

    user_input = st.session_state[
        "auto_question"
    ]

    del st.session_state[
        "auto_question"
    ]

else:

    user_input = st.chat_input(
        "Ask your legal question..."
    )


# ============================================================
# MAIN CHAT
# ============================================================

if user_input:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            user_input
        )


    # --------------------------------------------------------
    # TRANSLATE QUESTION TO ENGLISH
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="⚖️"
    ):

        with st.spinner(
            "🌍 Understanding your question..."
        ):

            try:

                if lang_code == "en":

                    question_en = user_input

                else:

                    question_en = translate_with_ollama(
                        user_input,
                        "en"
                    )

            except Exception as e:

                st.warning(
                    "⚠️ Translation failed. "
                    "Please try again."
                )

                question_en = user_input


        # ----------------------------------------------------
        # FAISS SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching legal knowledge..."
        ):

            if (
                index is not None
                and len(documents) > 0
            ):

                q_embed = embedding_model.encode(
                    [question_en]
                )

                q_embed = np.array(
                    q_embed
                ).astype(
                    "float32"
                )

                k = min(
                    3,
                    len(documents)
                )

                distances, indices = index.search(
                    q_embed,
                    k
                )

                context_parts = []

                for i in indices[0]:

                    if 0 <= i < len(documents):

                        context_parts.append(
                            documents[i]
                        )

                context = "\n".join(
                    context_parts
                )

            else:

                context = ""


        # ----------------------------------------------------
        # LEGAL PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are LegalAI, an AI Legal Knowledge Assistant specializing
in Indian law and the Constitution of India.

Answer the user's question clearly and accurately.

Use the legal knowledge provided below whenever relevant.

If the context is insufficient, use your general knowledge.

Keep the answer easy to understand.

Use:
- Short paragraphs
- Bullet points
- Simple examples
- Important legal terms

Do not provide personalized legal advice.
Do not pretend to be a lawyer.

LEGAL KNOWLEDGE:
{context}

USER QUESTION:
{question_en}

ANSWER:
"""


        # ----------------------------------------------------
        # ASK OLLAMA
        # ----------------------------------------------------

        with st.spinner(
            "🧠 LegalAI is thinking..."
        ):

            try:

                answer_en = ask_ollama(
                    prompt
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to Ollama."
                )

                st.info(
                    "Open another Command Prompt and run: "
                    "`ollama serve`"
                )

                answer_en = (
                    "Ollama server is not running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏳ Ollama took too long to respond."
                )

                answer_en = (
                    "The AI model took too long to respond."
                )

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )

                answer_en = (
                    "Unable to generate the response."
                )


        # ----------------------------------------------------
        # TRANSLATE ANSWER
        # ----------------------------------------------------

        if lang_code == "en":

            final_answer = answer_en

        else:

            with st.spinner(
                f"🌍 Translating to {selected_language}..."
            ):

                try:

                    final_answer = translate_with_ollama(
                        answer_en,
                        lang_code
                    )

                except Exception:

                    final_answer = answer_en

                    st.warning(
                        "⚠️ Translation failed. "
                        "Showing the English response."
                    )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.markdown(
            "### ⚖️ AI Legal Response"
        )

        st.markdown(
            final_answer
        )


        # ----------------------------------------------------
        # LEGAL DIAGRAM
        # ----------------------------------------------------

        show_legal_diagram(
            user_input
        )


        # ----------------------------------------------------
        # RETRIEVED CONTEXT
        # ----------------------------------------------------

        with st.expander(
            "📖 View Retrieved Legal Context"
        ):

            if context:

                st.write(
                    context
                )

            else:

                st.write(
                    "No specific legal context was retrieved."
                )


        # ----------------------------------------------------
        # DISCLAIMER
        # ----------------------------------------------------

        with st.expander(
            "⚠️ Legal Disclaimer"
        ):

            st.info(
                "This chatbot provides general legal information "
                "for educational purposes only. It is not professional "
                "legal advice."
            )


        # ----------------------------------------------------
        # EDGE TTS
        # ----------------------------------------------------

        try:

            audio_file = (
                f"response_{uuid.uuid4().hex}.mp3"
            )

            voice_map = {

                "en":
                    "en-IN-NeerjaNeural",

                "hi":
                    "hi-IN-SwaraNeural",

                "kn":
                    "kn-IN-SapnaNeural"
            }

            selected_voice = voice_map.get(
                lang_code,
                "en-IN-NeerjaNeural"
            )


            async def generate_voice():

                communicate = edge_tts.Communicate(

                    final_answer,

                    selected_voice,

                    rate=voice_rate,

                    volume="+0%"

                )

                await communicate.save(
                    audio_file
                )


            asyncio.run(
                generate_voice()
            )


            st.markdown(
                "#### 🔊 Listen to the answer"
            )

            st.audio(
                audio_file,
                format="audio/mp3"
            )


        except Exception:

            st.warning(
                "🔊 Voice output is currently unavailable."
            )


        # ----------------------------------------------------
        # SAVE AI RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer
            }
        )


# ============================================================
# CONSTITUTION EXPLORER
# ============================================================

st.divider()

st.markdown(
    "## 📜 Constitution Explorer"
)

st.caption(
    "Explore commonly asked questions about the Constitution "
    "of India."
)


constitutional_examples = [

    "What is Article 21 in simple words?",

    "Explain Article 19 with example",

    "What is Habeas Corpus?",

    "What is 42nd Amendment?",

    "What are Fundamental Rights?",

    "Difference between Rights and Duties"

]


cols = st.columns(2)


for i, q in enumerate(
    constitutional_examples
):

    if cols[i % 2].button(
        q,
        key="constitution_" + q,
        use_container_width=True
    ):

        st.session_state[
            "constitution_question"
        ] = q

        st.rerun()


# ============================================================
# CONSTITUTION SEARCH
# ============================================================

constitution_input = st.text_input(
    "🔎 Search Constitution"
)


if "constitution_question" in st.session_state:

    constitution_input = (
        st.session_state[
            "constitution_question"
        ]
    )

    del st.session_state[
        "constitution_question"
    ]


# ============================================================
# CONSTITUTION ANSWER
# ============================================================

if constitution_input:

    with st.spinner(
        "📖 Searching constitutional knowledge..."
    ):

        try:

            answer = ask_ollama(

                f"""
You are an Indian Constitution knowledge assistant.

Explain the following question in simple language.

Question:
{constitution_input}

Give:

1. A simple definition
2. Important points
3. A simple example if useful

Do not provide personalized legal advice.

Answer:
"""
            )

        except requests.exceptions.ConnectionError:

            answer = (
                "❌ Ollama is not running. "
                "Please run `ollama serve`."
            )

        except Exception as e:

            answer = (
                f"❌ Error: {str(e)}"
            )


    # --------------------------------------------------------
    # TRANSLATE CONSTITUTION ANSWER
    # --------------------------------------------------------

    if lang_code != "en":

        try:

            answer = translate_with_ollama(
                answer,
                lang_code
            )

        except Exception:

            st.warning(
                "⚠️ Translation failed. "
                "Showing English response."
            )


    st.markdown(
        "### 📜 Answer"
    )

    st.markdown(
        answer
    )


    show_legal_diagram(
        constitution_input
    )


    with st.expander(
        "⚠️ Legal Disclaimer"
    ):

        st.info(
            "This information is provided for general "
            "educational and legal awareness purposes only."
        )


    # --------------------------------------------------------
    # CONSTITUTION VOICE
    # --------------------------------------------------------

    try:

        audio_file = (
            f"constitution_{uuid.uuid4().hex}.mp3"
        )

        voice_map = {

            "en":
                "en-IN-NeerjaNeural",

            "hi":
                "hi-IN-SwaraNeural",

            "kn":
                "kn-IN-SapnaNeural"
        }

        selected_voice = voice_map.get(
            lang_code,
            "en-IN-NeerjaNeural"
        )


        async def generate_constitution_voice():

            communicate = edge_tts.Communicate(

                answer,

                selected_voice,

                rate=voice_rate,

                volume="+0%"

            )

            await communicate.save(
                audio_file
            )


        asyncio.run(
            generate_constitution_voice()
        )


        st.markdown(
            "#### 🔊 Listen"
        )

        st.audio(
            audio_file,
            format="audio/mp3"
        )


    except Exception:

        st.warning(
            "🔊 Voice output unavailable."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">

⚖️ <b>LegalAI</b><br><br>

Indian Legal Knowledge Assistant<br><br>

FAISS • Sentence Transformers • Ollama • Edge TTS • Streamlit

</div>
""", unsafe_allow_html=True)