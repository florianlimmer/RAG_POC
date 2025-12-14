import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Finde die Knowledge Base
KB_PATH = "knowledge_base"
DB_PATH = "vector_db/knowledge_store"

def index_knowledge_base():
    print(f"--- Indexiere Wissensdatenbank aus: {KB_PATH} ---")
    
    documents = []
    
    # 1. ALLE .txt Dateien finden (Batch Processing)
    txt_files = glob.glob(os.path.join(KB_PATH, "*.txt"))
    
    if not txt_files:
        print("❌ Keine Textdateien gefunden.")
        return

    # 2. Jede Datei laden
    for file_path in txt_files:
        print(f"📄 Lade Datei: {os.path.basename(file_path)}")
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {file_path}: {e}")

    # 3. Splitten (Chunking)
    print("✂️  Splitte Dokumente in Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    print(f"   -> {len(splits)} Chunks aus {len(txt_files)} Dateien erstellt.")

    # 4. Speichern (Vektor-DB)
    print("💾 Speichere in ChromaDB...")
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    
    # Alte DB überschreiben oder erweitern? Hier: Neu erstellen für Sauberkeit
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH) # Vorsicht: Löscht alte DB!
        
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print("✅ Indexierung abgeschlossen!")

if __name__ == "__main__":
    index_knowledge_base()