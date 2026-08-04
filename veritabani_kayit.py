import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Python içinde doğrudan çalışan yerel model (Ollama sunucusuna ihtiyaç duymaz)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Metinleri parçalara ayırma
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_texts = []

for filename in os.listdir("."):
    if filename.endswith(".txt"):
        with open(filename, 'r', encoding='utf-8') as f:
            chunks = text_splitter.split_text(f.read())
            all_texts.extend(chunks)

# Chroma veritabanına kaydetme
db = Chroma.from_texts(all_texts, embeddings, persist_directory="./cilt_bakimi_db")
print("HARİKA! Veritabanı başarıyla oluşturuldu.")