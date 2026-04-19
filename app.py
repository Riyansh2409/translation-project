import gradio as gr
from transformers import MarianMTModel, MarianTokenizer
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
loaded_models = {}

# Hindi
hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
hi_tokenizer = MarianTokenizer.from_pretrained(hi_model_name)
hi_model = MarianMTModel.from_pretrained(hi_model_name).to(device)
loaded_models["hindi"] = ("marian", hi_tokenizer, hi_model)

# Kannada
kn_model_name = "facebook/nllb-200-distilled-600M"
kn_tokenizer = NllbTokenizer.from_pretrained(kn_model_name)
kn_model = AutoModelForSeq2SeqLM.from_pretrained(kn_model_name).to(device)
loaded_models["kannada"] = ("nllb", kn_tokenizer, kn_model)

# Function
def translate(text, lang):
    if lang not in loaded_models:
        return "Invalid language"

    model_type, tokenizer, model = loaded_models[lang]

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

# UI
interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Enter English Text"),
        gr.Dropdown(["hindi", "kannada"], label="Select Language")
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="🌍 AI Translator (Hindi + Kannada)",
    description="Enter English text and translate into Hindi or Kannada"
)

interface.launch()