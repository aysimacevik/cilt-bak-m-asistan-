import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Cilt Bakımı Akıllı Asistanı", page_icon="✨", layout="wide"
)

# Profesyonel Cilt Bakım Teması ve Yazı Rengi Düzeltmesi
st.markdown(
    """
    <style>
    /* Genel Arka Plan ve Yazı Rengi */
    .stApp {
        background-color: #F4F7F6;
        color: #2C3E50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Yan Panel (Sidebar) Tasarımı */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Sidebar İçindeki Yazılar */
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #2C3E50 !important;
    }
    
    /* Başlıklar */
    h1 {
        color: #1A365D;
        font-weight: 700;
    }
    
    /* Sohbet Mesajlarının Okunabilirliği İçin Kesin Çözüm */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
        color: #2C3E50 !important;
    }
    
    /* Buton Tasarımları */
    .stButton > button {
        background-color: #E6FFFA;
        color: #234E52;
        border: 1px solid #B2F5EA;
        border-radius: 8px;
        width: 100%;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #319795;
        color: #FFFFFF;
        border-color: #319795;
    }
    
    /* Sohbet Girdi Kutusu */
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        color: #2C3E50 !important;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Ana Başlık Alanı
st.markdown(
    "<h1 style='text-align: center;'>✨ Cilt Bakımı Akıllı Asistanı</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #4A5568;'>Bilimsel makaleler ve"
    " rehberler ile desteklenen akıllı cilt bakım danışmanı.</p>",
    unsafe_allow_html=True,
)
st.divider()

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
  st.image(
      "https://images.unsplash.com/photo-1570172619644-dfd03ed5d881?w=400&auto=format&fit=crop&q=60",
      use_container_width=True,
  )
  st.markdown("### 🌿 Asistan Hakkında")
  st.info(
      "Bu sistem; cilt tipleri, deri yapısı ve bakım rehberlerini tarayarak"
      " tamamen yerel ve güvenli yanıtlar veren yapay zeka destekli bir RAG"
      " uygulamasıdır."
  )

  st.markdown("---")
  st.markdown("### 💡 Hızlı Sorular")
  st.markdown("Merak ettiklerini hemen test etmek için tıkla:")

  if "quick_question" not in st.session_state:
    st.session_state.quick_question = None

  if st.button("🌸 Kuru cilt özellikleri nelerdir?"):
    st.session_state.quick_question = "Kuru cilt özellikleri nelerdir?"
  if st.button("💧 Evde kurulama testi nasıl yapılır?"):
    st.session_state.quick_question = "Evde kurulama testi nasıl yapılır?"
  if st.button("⚖️ Karma cilt tipi nasıl anlaşılır?"):
    st.session_state.quick_question = "Karma cilt tipi nasıl anlaşılır?"


# Veritabanı ve Model Ayarları
@st.cache_resource
def veritabanini_yukle():
  try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="./cilt_bakimi_db", embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": 2})
  except Exception as e:
    return None


retriever = veritabanini_yukle()

try:
  llm = ChatOllama(model="llama3.2:3b", temperature=0.0)
except Exception as e:
  st.error(f"Ollama başlatılamadı: {e}")

template = """Aşağıdaki bağlamı (makale ve rehber içeriklerini) kullanarak soruya net, Türkçe ve doğru bir yanıt ver. 
Eğer cevap bağlamda yoksa, kafadan sallama, sadece bağlamdakileri söyle.

Bağlam:
{context}

Soru: {question}

Cevap:"""

prompt_template = ChatPromptTemplate.from_template(template)

user_input = st.chat_input("Cilt bakımı hakkında ne öğrenmek istiyorsun?")

if st.session_state.quick_question:
  user_question = st.session_state.quick_question
  st.session_state.quick_question = None
else:
  user_question = user_input

if user_question:
  with st.chat_message("user"):
    st.write(user_question)

  with st.chat_message("assistant"):
    with st.spinner("Makaleler taranıyor..."):
      try:
        if retriever is not None:
          docs = retriever.invoke(user_question)
          context = "\n".join([doc.page_content for doc in docs])
        else:
          context = "Veritabanı bulunamadı."

        full_prompt = prompt_template.format(
            context=context, question=user_question
        )
        response = llm.invoke(full_prompt)
        answer_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        st.write(answer_text)
      except Exception as e:
        st.error(f"Hata oluştu: {e}")