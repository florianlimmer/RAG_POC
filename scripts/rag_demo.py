import os
import pandas as pd
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- TEIL 1: AUTENTISIERUNGSSCHICHT ---
class AuthenticationLayer:
    """Simuliert eine sichere, separate Benutzerdatenbank."""
    def __init__(self, df):
        # Wir hashen das Passwort nicht (da es nur ein Prototyp ist),
        # speichern es aber separat vom RAG-Kontext ab.
        self.auth_db = df[['Kundennummer', 'Passwort']].set_index('Kundennummer').to_dict('index')

    def check_auth(self, kundennummer: str, passwort: str) -> bool:
        try:
            # Konvertiere Kundennummer in String und bereinige Eingaben
            kundennummer = str(kundennummer).strip()
            passwort_input = str(passwort).strip()
            
            # Hole gespeichertes Passwort
            entry = self.auth_db.get(kundennummer)
            if not entry:
                return False
                
            stored_password = str(entry.get('Passwort')).strip()
            
            # Debugging (kann später entfernt werden)
            # print(f"Check: {kundennummer} | Input: {passwort_input} | Stored: {stored_password}")
            
            return stored_password == passwort_input
        except Exception as e:
            print(f"Auth Fehler: {e}")
            return False

# --- DATEN LADEN ---

# Pfad-Konstruktion: Geht davon aus, dass wir in scripts/rag_application sind
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # RAG-Bank-PoC Root
if not os.path.basename(base_dir) == "RAG-Bank-PoC": 
    # Fallback falls Skript direkt aus Root gestartet wird
    base_dir = "."

data_dir = os.path.join(base_dir, "knowledge_base")
csv_path = os.path.join(data_dir, "bank_daten.csv")

try:
    # CSV laden
    df = pd.read_csv(csv_path, sep=",", skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    # WICHTIG: Kundennummer und Passwort explizit zu Strings machen
    if 'Kundennummer' in df.columns:
        df['Kundennummer'] = df['Kundennummer'].astype(str).str.strip()
    if 'Passwort' in df.columns:
        df['Passwort'] = df['Passwort'].astype(str).str.strip()
        
except FileNotFoundError:
    print(f"❌ Fehler: '{csv_path}' nicht gefunden. Bitte Pfad prüfen oder Generator ausführen!")
    sys.exit(1)


# Trennung für Auth Layer
auth_df = df.copy() 
auth_system = AuthenticationLayer(auth_df)

# RAG Vorbereitung (Passwort entfernen)
if 'Passwort' in df.columns:
    df_rag_ready = df.drop(columns=['Passwort'])
else:
    df_rag_ready = df.copy()

# Funktion: Row to Text
def row_to_text(row):
    return (
        f"Kunde {row['Vorname']} {row['Nachname']} (Kundennummer: {row['Kundennummer']}, geboren am:{row['Geburtsdatum']}) "
        f"hat ein {row['Kontoart']} mit der IBAN {row['IBAN']}. "
        f"Kontostand: {row['Kontostand']} EUR. Dispo-Limit: {row['Dispo_Limit']} EUR. "
        f"Genossenschaftsmitglied: {row['Ist_Mitglied']}. "
        f"Sicherheitsverfahren: {row['TAN_Verfahren']}. Risikoklasse: {row['Risikoklasse']}."
    )

df_rag_ready["rag_text"] = df_rag_ready.apply(row_to_text, axis=1)
documents = df_rag_ready["rag_text"].tolist()
ids = [str(i) for i in range(len(df_rag_ready))]
metadatas = [{'kundennummer': str(k)} for k in df_rag_ready['Kundennummer']]

print(f"   ✅ {len(documents)} Kundendatensätze geladen.")


# --- TEIL 2: DATENBANKEN ---

# A. WISSENSDATENBANK (LangChain)
KNOWLEDGE_DB_PATH = os.path.join(base_dir, "vector_db", "knowledge_store")
knowledge_retriever = None

if os.path.exists(KNOWLEDGE_DB_PATH):
    embedding_function = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    # Versuche DB zu laden
    try:
        knowledge_vectorstore = Chroma(persist_directory=KNOWLEDGE_DB_PATH, embedding_function=embedding_function)
        # Hier wird festgelegt, wie viele retrieval chunks dem LLM zur Verfügung gestellt werden
        knowledge_retriever = knowledge_vectorstore.as_retriever(search_kwargs={"k": 4})
        print("   -> Wissensdatenbank geladen.")
    except Exception as e:
        print(f"⚠️  Fehler beim Laden der Wissens-DB: {e}")
else:
    print(f"⚠️  Keine Wissensdatenbank unter {KNOWLEDGE_DB_PATH} gefunden.")

# B. KUNDEN-DATENBANK (In-Memory Chroma)
print("2. Erstelle Kunden-Index...")
embed_model = SentenceTransformer('intfloat/multilingual-e5-base')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="bank_data")
embeddings = embed_model.encode(documents)
collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
print("   -> Kunden-Datenbank bereit!")


# --- TEIL 3: LLM LADEN ---

print("3. Lade LLM (Mistral/OpenVINO)...")
model_id = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov" 

try:
    model = OVModelForCausalLM.from_pretrained(
        model_id, export=False, device="GPU", 
        ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "./model_cache"}
    )
    model._supports_cache_class = False 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
except Exception as e:
    print(f"❌ Fehler beim Laden des Modells: {e}")
    sys.exit(1)

# --- TEIL 4: DIE RAG FUNKTION ---
# --- TEIL 4: RAG FUNKTION ---

def ask_bank_bot(question: str, logged_in_kundennummer: str) -> str:
    
    # 1. WISSENSDATENBANK RETRIEVAL
    knowledge_context = "Keine allgemeinen Infos verfügbar."
    if knowledge_retriever:
        docs = knowledge_retriever.invoke(question)
        if docs:
            print(f"\n📚 [RETRIEVAL] Gefundene Knowledge-Chunks ({len(docs)}):")
            knowledge_parts = []
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get('source', 'Unbekannt'))
                content = doc.page_content.replace('\n', ' ')
                print(f"   🔹 Chunk {i+1} (aus {source}): \"{content[:100]}...\"") # Zeigt die ersten 100 Zeichen
                knowledge_parts.append(f"Quelle {source}: {doc.page_content}")
            knowledge_context = "\n\n".join(knowledge_parts)
        else:
            print("\n📚 [RETRIEVAL] Keine passenden Knowledge-Einträge gefunden.")

# 2. KUNDENDATEN RETRIEVAL
    customer_context = "Keine Kundendaten gefunden."
    results = collection.query(
        query_embeddings=embed_model.encode([question]), 
        n_results=1, # Nur den exakten Kunden
        where={"kundennummer": logged_in_kundennummer}
    )
    
    if results['documents'] and results['documents'][0]:
        found_data = results['documents'][0][0]
        print(f"\n👤 [RETRIEVAL] Gefundene Kundendaten:")
        print(f"   🔹 \"{found_data}\"")
        customer_context = found_data
    else:
        print(f"\n👤 [RETRIEVAL] ⚠️ Keine Daten für Kunde {logged_in_kundennummer} gefunden!")

    print("-" * 60)

    # 3. PROMPT ERSTELLEN
    prompt = f"""[INSTRUKTION]
Du bist ein hilfreicher und höflicher Bank-Assistent der Atruvia AG.
Nutze die Wissensdatenbank und die Kundendaten, um Kundenfragen zu beantworten.
Beachte die Informationen aus der Wissensdatenbank (insbesondere die AGBs) strikt. 
Wenn die Antwort nicht in der Wissensdatenbank oder den Kundendaten zu finden ist, antworte mit "Hierzu habe ich keine verlässlichen Informationen."
Antworte ausschließlich in Deutsch.

[WISSENSDATENBANK]
{knowledge_context}

[KUNDENDATEN]
{customer_context}

[FRAGE]
{question}
"""
    
    # 4. GENERIERUNG
    messages = [{"role": "user", "content": prompt}]
    outputs = pipe(messages, do_sample=False)
    return outputs[0]['generated_text'][-1]['content']

# --- TEIL 5: INTERACTION LOOP ---

print("\n✅ SYSTEM BEREIT! Bitte anmelden.")

while True:
    k_nr = input("\n🔑 Kundennummer (oder 'q'): ")
    if k_nr.lower() in ['q', 'logout', 'exit']: break
    
    pw = input("🔒 Passwort: ")
    
    if auth_system.check_auth(k_nr, pw):
        # Kundennamen aus der Datenbank abrufen
        kundennummer_clean = str(k_nr).strip()
        customer_row = df[df['Kundennummer'] == kundennummer_clean]
        if not customer_row.empty:
            vorname = customer_row.iloc[0]['Vorname']
            nachname = customer_row.iloc[0]['Nachname']
            print(f"\n✅ Anmeldung erfolgreich. Willkommen, {vorname} {nachname}! Sie können nun Fragen stellen.")
        else:
            print("\n✅ Anmeldung erfolgreich. Sie können nun Fragen stellen.")
        
        while True:
            frage = input(f"\n💬 Frage (Kunde {k_nr}): ")
            if frage.lower() in ['q', 'logout', 'exit']:
                print("--- Logout ---")
                break
                
            print("⏳ ...")
            antwort = ask_bank_bot(frage, str(k_nr).strip())
            print(f"🤖: {antwort}")
            print("-" * 40)
    else:
        print("❌ Login fehlgeschlagen.")