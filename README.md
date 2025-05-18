# RAG-cross-LLM-docbot

## Overview

This project is a command-line document assistant powered by RAG (Retrieval-Augmented Generation) and multiple LLMs.  
It solves the problem of quickly retrieving and synthesizing insights from specific PDF documents by combining semantic search, FAISS vector search, and large language models via API calls.  
Designed modular Python backend with LangChain, document chunking, vector embedding, and retrieval capabilities. It can be easily extended to support various LLMs and document formats.  

**Target audience**: Students, engineers, and companies needing internal knowledge search automation.  
**Key functionality**: Document retrieval, multi-LLM query resolution, API integration.

---

## Features

- Document ingestion and vector-based retrieval using FAISS.
- Query resolution using OpenAI API, Hugging Face LLMs.
- Custom document-based RAG prompts (context + question).
- Mock automation triggers (e.g., generate ticket on matched answer).
- Error handling for missing documents or vector index.
- Easy to extend with new LLM providers or file formats.
- Modular design for easy integration with other systems.

---

## Tech Stack

| Category        | Technologies                         |
|----------------|--------------------------------------|
| Backend         | Python (LangChain, Transformers)     |
| Embeddings      | `sentence-transformers`              |
| Retrieval       | FAISS                                |
| API / LLMs      | OpenAI API, Hugging Face Transformers|
| Testing         | Manual / CLI-based testing           |
| Dev Tools       | Git, dotenv, PyMuPDF, VS Code        |

---

## Lesson Learned

This project has improved my skills in the following areas :

- Building modular RAG pipelines using LangChain and FAISS.
- Switching dynamically between LLM providers (OpenAI, Hugging Face).
- Handling missing dependencies and graceful CLI UX.
- Designing retrieval prompts for accuracy and transparency.

---

## Try it out

### Prerequisites
- Python 3.10 or higher  
- OpenAI API key (OpenAI account)  
- `documents/` directory in the root of the project with PDF files to be ingested.  

### Quickstart
```bash
# Clone the repo
git clone https://github.com/svvoii/RAG-cross-LLM-docbot
cd RAG-cross-LLM-docbot

# Adding your OpenAI API key
touch .env
echo "OPENAI_API_KEY=your_openai_api_key" > .env

# set up the virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ingest documents
python process_docs.py

# Run the CLI
python main.py

```

### Usage

- Add your PDF documents to the `documents/` directory.  
- Run `python process_docs.py` to ingest the documents and create `retriever.pkl` file.  
- Run `python main.py` to interact with the docbot via CLI.  
- Type your question.  
- Choose the LLM model.  

---

## Author

[My GitHub](https://github.com/svvoii)  
[My LinkedIn](https://www.linkedin.com/in/bocancia/)  
[My Portfolio](https://sbocanci.me/)  

---
