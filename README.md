# RAG_POC - Bank-Anwendung Proof of Concept

Ein vollständiger RAG (Retrieval-Augmented Generation) Proof-of-Concept für eine Genossenschaftsbank mit synthetischer Datengenerierung, AGB-Erstellung, Wissensdatenbank-Embedding und interaktivem Chatbot.

## 📋 Übersicht

Dieses Projekt simuliert eine vollständige KI-Pipeline für eine Genossenschaftsbank:

- **🔐 Privacy First:** Lokale Ausführung aller LLMs (kein Datenabfluss an Cloud-Provider).
- **🛡️ Zugriffskontrolle:** Authentifizierter Chatbot, der Kundendaten strikt filtert (Metadaten-Filterung).
- **🧠 Hybrid RAG:** Kombiniert strukturierte Kundendaten (CSV) mit unstrukturiertem Unternehmenswissen (AGB, Historie).
- **⚙️ Hardware-Optimiert:** Nutzt Intel OpenVINO für performante Inferenz auf Standard-Hardware.

## 🏗️ Architektur

```
RAG-Bank-PoC/
├── knowledge_base/            
│   ├── bank_daten.csv                 # Strukturierte Kundendaten
│   ├── agb_regeln.txt                 # AGBs
│   ├── historie.txt                   # https://atruvia.de/unternehmen/unternehmensgruppe/geschichte
│   └── aktuelles.txt                  # https://atruvia.de/unternehmen/wir-sind-atruvia/zahlen-und-fakten
├── vector_db/                         # ChromaDB Speicher
├── scripts/
|   ├── agb_generator.py               # Erstellt AGBs mit LLM call
|   |── knowledge_indexer.py           # Erstellt die Embedding Datenbank
|   |── rag_demo.py                    # Demo
|   └── synt_data_generator.py         # Erstellt synthetische Kundendaten mit LLM call
└── pyproject.toml                     # Abhängigkeitsmanagement (uv)
```

### Technologie-Stack

- **LLM**: Mistral-7B-Instruct-v0.2 (optimiert für Intel/OpenVINO, INT4 quantisiert)
- **RAG Framework**: **LangChain** (langchain-huggingface, langchain-chroma) für Knowledge Base Retrieval
- **Embedding Models**: 
  - `intfloat/multilingual-e5-base` für Knowledge Base (LangChain)
  - `intfloat/multilingual-e5-base` für Kundendaten (Sentence Transformers)
- **Vector Databases**: 
  - ChromaDB (persistent) für Knowledge Base (AGB, Unternehmensinfos)
  - ChromaDB (In-Memory) für Kundendaten mit Metadaten-Filterung
- **Architektur**: **Zweistufige RAG** - Knowledge Base + Kundendaten
- **Authentifizierung**: Separate Passwort-Datenbank
- **RAG**: Kundenspezifische Filterung mit Metadaten

## 🚀 Schnellstart

### Voraussetzungen

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) Package Manager
- Hugging Face Account mit Token
- Intel GPU (optional, für OpenVINO Beschleunigung)

### Installation

1. **Umgebungsvariablen einrichten:**

   Erstelle eine `.env` Datei im Projektroot (`RAG/`) mit deinem Hugging Face Token:
   ```
   HF_TOKEN=dein_huggingface_token_hier
   ```

2. **Abhängigkeiten installieren:**

   ```bash
   cd RAG
   uv sync
   ```

3. **Virtuelle Umgebung aktivieren:**

   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

## 📝 Verwendung

### Schritt 1: Synthetische Bankdaten generieren

Generiert realistische Testdaten für Bankkunden mit 35 Datensätzen:

```bash
uv run RAG_POC/scripts/synt_data_generator.py
```

**Output:** `RAG_POC/data/bank_daten.csv`

**Enthaltene Spalten:**
- `Kundennummer` - 8-stellige eindeutige Nummer
- `Nachname`, `Vorname` - Kundenname
- `Passwort` - 4-stelliges Passwort für Authentifizierung
- `Geburtsdatum` - Format: YYYY-MM-DD
- `Kontoart` - Girokonto, Sparkonto, Tagesgeld, Festgeld
- `IBAN` - Deutsche IBAN (DE...)
- `Kontostand` - Mit 2 Dezimalstellen
- `Dispo_Limit` - Überziehungslimit
- `Ist_Mitglied` - ja/nein
- `Anzahl_Genossenschaftsanteile` - Anzahl der Anteile
- `TAN_Verfahren` - PIN, TAN, Biometrie
- `Risikoklasse` - 1, 2, 3, 4, 5

### Schritt 2: AGB-Regeln generieren

Erstellt Geschäftsbedingungen für Genossenschaftsanteile basierend auf definierten Regeln:

```bash
uv run RAG_POC/scripts/agb_generator.py
```

**Output:** `RAG_POC/data/agb_regeln.txt`

**Enthaltene Regeln:**
- Risikoklassifizierung: Nur Kunden der Risikoklasse 1, 2 oder 3 dürfen Anteile erwerben
- Preisgestaltung: 500€ pro Anteil, nur ganze Anteile
- Altersbeschränkungen:
  - Vor 1990 geboren: maximal 10 Anteile
  - Nach 1990 geboren: maximal 5 Anteile


### Schritt 3: Wissensdatenbank indexieren (Embedding)

```bash
uv run RAG_POC/scripts/knowledge_indexer.py
```

### Schritt 4: RAG-Demo starten

Interaktive Demo mit Authentifizierung und kundenspezifischen Antworten:

```bash
uv run RAG_POC/scripts/rag_demo.py
```

## 💬 Interaktive Demo - Verwendung

### Authentifizierung

Beim Start des Skripts werden Sie zur Anmeldung aufgefordert:

```
🔑 Kundennummer (oder 'exit' zum Beenden): 1
🔒 Passwort (oder 'exit' zum Beenden): 1234

✅ Anmeldung erfolgreich. Willkommen, Max Müller! Sie können nun Fragen stellen.
```

**Programm beenden während der Authentifizierung:**
- Eingabe von `exit`, `quit`, `q` oder `beenden`

### Fragen stellen

Nach erfolgreicher Anmeldung können Sie Fragen zu Ihren Kontodaten stellen:

```
💬 Frage zur Kundendatenbank (oder 'logout'): Wie hoch ist mein Kontostand?
🤖 Bot: Ihr aktueller Kontostand beträgt 5000.50 EUR.
```

**Beispiel-Fragen:**
- "Wie hoch ist mein Kontostand?"
- "Darf ich Genossenschaftsanteile kaufen?"
- "Wie viele Anteile kann ich maximal erwerben?"
- "Wann wurde die Atruvia AG gegündet?"
- "Welchen Umsatz hatte die Atruvia AG im Jahr 2024?"

**Abmelden:**
- Eingabe von `logout` oder `q`

## 🔐 Sicherheitsfeatures

### Datenschutz

- **Separate Authentifizierung**: Passwörter werden getrennt von RAG-Daten gespeichert.
- **Metadaten-Filterung**: Jeder Benutzer sieht nur seine eigenen Daten.
- **Keine Passwort-Exposition**: Passwörter werden nicht in den RAG-Texten gespeichert.
- **Lokale Datenhaltung**: Keine Daten verlassen das System (keine OpenAI API Aufrufe), was DSGVO-Konformität erleichtert.

### Datenfluss

1. **Indexierung:**
   - CSV-Daten werden in natürliche Sprache umgewandelt
   - Embeddings werden mit Sentence Transformers erstellt
   - Dokumente werden in ChromaDB mit Metadaten (Kundennummer) gespeichert

2. **Query-Verarbeitung:**
   - Benutzerfrage wird in Embedding umgewandelt
   - Ähnlichkeitssuche in ChromaDB mit Metadaten-Filter (nur eigene Daten)
   - Relevante Stellen aus der indexierten Wissensdatenbank werden als zusätzlicher Context verwendet
   - LLM generiert Antwort basierend auf Context und Kundendaten



### Erweiterungsmöglichkeiten

- **Passwort-Hashing**: Sichere Passwort-Speicherung implementieren
- **Web-Interface**: Gradio UI erstellen


