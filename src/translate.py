from transformers import MarianMTModel, MarianTokenizer
import torch

# Model name (English → Hindi)
model_name = "Helsinki-NLP/opus-mt-en-hi"

# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Translation function
def translate(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

# Test sentences
sentences = [
    "Hello, how are you?",
    "I am learning machine learning.",
    "Transformers are very powerful."
]

# Run translation
results = translate(sentences)

# Print results
for en, hi in zip(sentences, results):
    print(f"EN: {en}")
    print(f"HI: {hi}")
    print("-" * 50)