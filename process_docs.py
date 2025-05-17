# PDF parsing

import os
import faiss
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


DOCS_PATH = "documents"

# Processes documents into chunks and creates a FAISS index for retrieval (retriever.pkl)
def main():
    if not os.path.exists(DOCS_PATH):
        print(f"Directory `{DOCS_PATH}` does not exist. Please create it and add PDF files.")
        return

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in documents/ folder. Please add relevant PDF files.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    
    for filename in pdf_files:
        print(f"Processing {filename}...")
        loader = PyMuPDFLoader(os.path.join(DOCS_PATH, filename))
        docs = loader.load()
        documents.extend(text_splitter.split_documents(docs))
    
    if not documents:
        print("No text could be extracted from the PDF files.")
        return
    
    texts = [doc.page_content for doc in documents]
    embeddings = model.encode(texts)
    
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)
    
    with open("retriever.pkl", "wb") as f:
        pickle.dump((index, texts), f)

    print(f"Processed {len(documents)} documents and saved to retriever.pkl")

    ## DEBUG ##
    print(f"Content of retriever.pkl: {texts[:3]}")  # Print first 3 texts for debugging
    # # # # # #


if __name__ == "__main__":
    main()
