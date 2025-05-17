# to retrieve context
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from LLM_handler import query_openai, query_huggingface

# mute huggingface warnings related to tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = {
    "1": ("GPT-4 (OpenAI)", query_openai, "gpt-4"),
    "2": ("GPT-3.5 (OpenAI)", query_openai, "gpt-3.5"),
    "3": ("GPT-2 (Hugging Face)", query_huggingface, "gpt2"),
    "4": ("DistilGPT-2 (Hugging Face)", query_huggingface, "distilgpt2"),
    # ADD MORE MODELS HERE
}

# Retrieves context based on input query with FAISS indexing
def retrieve_context(input_query, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open("retriever.pkl", "rb") as f:
        index, text = pickle.load(f)
    
    query_vector = model.encode([input_query])
    distances, indices = index.search(np.array(query_vector), top_k)

    ## DEBUG ##
    # print(f"\tDistances: {distances}")
    # print(f"\tIndices: {indices}")
    # print(f"\tTexts: ")
    # for i in indices[0]:
    #     print(f"\t\t{text[i]}")
    # # # # # #

    return [text[i] for i in indices[0]]


def create_prompt(input_query, context):
    prompt = f"""
    Context: {context}
    Question: {input_query}
    """
    return prompt


def main():
    if not os.path.exists("retriever.pkl"):
        print("No retriever.pkl found. Please run `process_docs.py` first..")
        return

    input_query = input("Ask your question: ")
    print(f"You asked: {input_query}")
    
    context = retrieve_context(input_query)
    prompt_to_process = create_prompt(input_query, context)

    print("Available models:")
    for key, (name, _, _) in MODELS.items():
        print(f"{key}: {name}") 

    model_choice = input(f"Choose a model (1-{len(MODELS)}): ").strip()

    if model_choice in MODELS:
        model_name, query_function, model_id = MODELS[model_choice]
        print(f"Using model: {model_name}...")
        answer = query_function(prompt_to_process, model=model_id)
    else:
        print("Invalid model choice. Please choose a valid model.")
        return

    print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
