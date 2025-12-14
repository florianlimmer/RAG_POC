import os
import dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM # Intel Optimierung

# Lade Umgebungsvariablen aus der .env Datei
userdata = dotenv.dotenv_values()
hf_token = userdata.get('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)

# Definiere das Modell
model_id = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov"

# Definiere den Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Lade das Modell
model = OVModelForCausalLM.from_pretrained(model_id)

# Fix für Kompatibilität mit transformers: Füge fehlendes Attribut hinzu
if not hasattr(model, '_supports_cache_class'):
    model._supports_cache_class = False

# Definiere die Pipeline, wir nutzen text-generation, da wir ein Text generieren wollen
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4000,
)

# Definiere den Prompt für die Datengenerierung
# Mistral erwartet nur einen User-Prompt
user_prompt = """
Du bist ein Experte für Bankdaten und Datengenerierung.
Erstelle realistische synthetische Testdaten für ein Core-Banking-System einer Genossenschaftsbank.
Achte auf logische Konsistenz (z.B. hat ein Kind unter 18 Jahren kein Dispo-Limit).
Antworte NUR mit dem CSV-Inhalt. Keine Einleitung, kein Markdown, kein Schlusswort, keine Anführungszeichen. 


Erstelle eine synthetische CSV-Datei für Bankdaten mit 35 Zeilen von Kundendaten.
Füge als erste Zeile die Spaltennamen hinzu.
Spalten: 
Kundennummer (zufällig generiert, 8-stellig), Nachname, Vorname, Passwort (zufällig generiert, 4-stellig), Geburtsdatum (YYYY-MM-DD), Kontoart (Girokonto, Sparkonto, Tagesgeld, Festgeld), IBAN (DE...), 
Kontostand (2 Dezimalstellen), Dispo_Limit, Ist_Mitglied (ja/nein), Anzahl_Genossenschaftsanteile, 
TAN_Verfahren (PIN, TAN, Biometrie), Risikoklasse (1, 2, 3, 4, 5).
"""

messages = [
    {"role": "user", "content": user_prompt},
]   

print("Generiere Daten...")

outputs = pipe(
    messages,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Extrahiere den generierten Text aus der Pipeline
generated_text = outputs[0]['generated_text'][-1]['content']

print("--- Generierte CSV Daten ---")
print(generated_text)

# Speichern der generierten Daten im data Ordner
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "bank_daten.csv")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\n✅ CSV-Datei gespeichert in '{output_path}'.")
