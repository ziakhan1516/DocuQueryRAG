# DocuQueryRAG

**DocuQueryRAG** is a Retrieval-Augmented Generation (RAG) system for question answering over documents (PDF, DOCX, PPTX, TXT). It uses sentence embeddings, FAISS vector search, and local LLMs (like Mistral via Ollama) to provide context-aware answers from uploaded files.


## Features

- Supports multiple file types: PDF, DOCX, PPTX, TXT
- Text chunking and semantic embedding with `all-MiniLM-L6-v2`
- Fast similarity search with FAISS
- Flexible LLM integration via Ollama (default: `mistral:7b`)
- Command-line option to switch models (`--model`) , but first you have to download that model from ollama site.
- Interactive web UI powered by Gradio

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DocuQueryRAG.git
cd DocuQueryRAG
```
## Install dependencies:
```bash
pip install -r requirements.txtt
```
## Run the app with the default Mistral model:
```bash
pip install -r requirements.txtt
python app.py --model mistral:7b
For changing the Model you just have to give the model name here.
```
