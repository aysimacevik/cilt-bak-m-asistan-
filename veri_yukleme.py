import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Klasör ayarı: Şu an zaten veri klasörünün içindeyiz, '.' bunu ifade eder
data_folder = "." 

# 2. Parçalayıcı ayarı: Metni 500 karakterlik parçalara böl, 50 karakterlik örtüşme olsun
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
print("Dosyalar aranıyor...")
# 3. Dosyaları tara ve işle
for filename in os.listdir(data_folder):
    # Sadece .txt dosyalarını al
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Metni parçalara böl
            chunks = text_splitter.split_text(content)
            print(f"--- {filename} ---")
            print(f"Toplam {len(chunks)} parçaya bölündü.\n")
            
            # Parçaların ilk 100 karakterini ekrana yazdır (Kontrol için)
            for i, chunk in enumerate(chunks):
                print(f"Parça {i+1}: {chunk[:100]}...")