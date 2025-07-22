# DocuQueryRAG

**DocuQueryRAG** is a Retrieval-Augmented Generation (RAG) system for question answering over documents (PDF, DOCX, PPTX, TXT). It uses sentence embeddings, FAISS vector search, and local LLMs (like Mistral via Ollama) to provide context-aware answers from uploaded files.

---

## Features

- Supports multiple file types: PDF, DOCX, PPTX, TXT
- Text chunking and semantic embedding with `all-MiniLM-L6-v2`
- Fast similarity search with FAISS
- Flexible LLM integration via Ollama (default: `mistral:7b`)
- Command-line option to switch models (`--model`)
- Interactive web UI powered by Gradio

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DocuQueryRAG.git
cd DocuQueryRAG
```

```bash
pip install -r requirements.txtt
```
