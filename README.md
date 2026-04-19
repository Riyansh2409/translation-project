# 🌍 AI Translator (English → Hindi & Kannada)

An AI-powered translation system that converts English text into Hindi and Kannada using Transformer-based models from Hugging Face.

---

## 🚀 Features

* 🔤 English → Hindi translation (MarianMT)
* 🌐 English → Kannada translation (NLLB - Meta AI)
* 🔁 Interactive loop for continuous input
* ✅ Input validation for language selection
* ⚡ GPU/CPU support using PyTorch

---

## 🧠 Models Used

| Language | Model                            | Type     |
| -------- | -------------------------------- | -------- |
| Hindi    | Helsinki-NLP/opus-mt-en-hi       | MarianMT |
| Kannada  | facebook/nllb-200-distilled-600M | NLLB     |

---

## 📂 Project Structure

```
translation-project/
│── src/
│   └── translate.py
│── app.py   (Gradio UI - optional)
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Riyansh2409/translation-project.git
cd translation-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python src/translate.py
```

---

## 💻 Example Usage

```
Enter English text: Hello
Choose language (hindi/kannada): hindi

Translated: नमस्ते
```

---

## ⚠️ Notes

* First run may take time due to model download (~2.5GB for NLLB)
* Translation may not always be perfectly natural (model limitation)
* Kannada translation uses multilingual NLLB model

---

## 🚀 Future Improvements

* 🌐 Web UI using Gradio
* 📊 Translation quality evaluation (BLEU score)
* 🔁 Support for more languages
* ☁️ Deployment (Hugging Face / Streamlit)

---

## 👨‍💻 Author

**Riyansh Jain**
B.Tech AIML Student

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
