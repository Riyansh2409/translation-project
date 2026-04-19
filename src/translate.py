from transformers import MarianMTModel, MarianTokenizer
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_models = {}

# Hindi (Marian)
hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
hi_tokenizer = MarianTokenizer.from_pretrained(hi_model_name)
hi_model = MarianMTModel.from_pretrained(hi_model_name).to(device)
loaded_models["hindi"] = ("marian", hi_tokenizer, hi_model)

# Kannada (NLLB)
kn_model_name = "facebook/nllb-200-distilled-600M"
kn_tokenizer = NllbTokenizer.from_pretrained(kn_model_name)
kn_model = AutoModelForSeq2SeqLM.from_pretrained(kn_model_name).to(device)
loaded_models["kannada"] = ("nllb", kn_tokenizer, kn_model)

# Translation function
def translate(text, target_lang):
    model_type, tokenizer, model = loaded_models[target_lang]

    if model_type == "marian":
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    elif model_type == "nllb":
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("kan_Knda")
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return "Translation error"
# Input
while True:
    text = input("\nEnter English text (type 'exit' to quit): ")

    if text.lower() == "exit":
        print("Exiting... 👋")
        break

    lang = input("Choose language (hindi/kannada): ").lower()

    # 🔥 YAHI PAR validation
    if lang not in loaded_models:
        print("Invalid language. Try again.")
        continue

    result = translate(text, lang)
    print("Translated:", result)