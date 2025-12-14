import os
import dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM

# Lade Umgebungsvariablen aus der .env Datei
userdata = dotenv.dotenv_values()
hf_token = userdata.get('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)

# Definiere das Modell
model_id = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov"

print(f"Lade {model_id} für AGB-Generierung...")

# Lade das Modell
model = OVModelForCausalLM.from_pretrained(
    model_id, export=False, device="GPU", 
    ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "./model_cache"}
)
model._supports_cache_class = False 

# Definiere den Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Definiere die Pipeline, wir nutzen text-generation, da wir ein Text generieren wollen
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# Definiere den Prompt für die AGB-Generierung
prompt_agb = """
[INSTRUKTION]
Du bist ein Jurist und erstellst AGB-Klauseln für eine Genossenschaftsbank.
Erstelle klare Klauseln zur Risikoklassifizierung in Bezug auf den Kauf von Gesellschaftsanteilen.
Jede Regel soll in einer eigenen Klausel formuliert werden.

Regel: Nur Kunden der Risikoklasse 1, 2 oder 3 dürfen Genossenschaftsanteile erwerben. 
Kunden der Risikoklasse 4 und 5 sind vom Kauf weiterer Anteile ausgeschlossen.
Es darf keine Ausnahme gemacht werden.

Regel: Genossenschaftsanteile kosten 500€ pro Anteil. Es können nur ganze Anteile gekauft werden.

Regel: Genossenschaftsmitglieder, die vor 1990 geboren sind, dürfen maximal 10 Anteile erwerben.

Regel: Genossenschaftsmitglieder, die nach 1990 geboren sind, dürfen maximal 5 Anteile erwerben.

Formuliere eine kleine Einleitung, die die AGB-Klauseln erklärt.

[AUFGABE]
Erstelle die AGB-Klauseln basierend auf der oben genannten Regel.
"""

messages = [{"role": "user", "content": prompt_agb}]   

print("Generiere AGB-Klausel...")

outputs = pipe(
    messages,
    do_sample=False,
    temperature=0.1, # Wichtig: Niedrige Temperatur, um genaue Regeln zu erhalten
)

agb_text = outputs[0]['generated_text'][-1]['content'].strip()

# Bereinigung des Texts
if "AGB" in agb_text[:50]:
    agb_text = agb_text.split('\n', 1)[-1].strip()

# Speichern der AGB in einer Textdatei im data Ordner
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "agb_regeln.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(agb_text)

print("\n--- Generierte AGB-Klausel ---")
print(agb_text)
print(f"\n✅ AGB-Klausel gespeichert in '{output_path}'.")