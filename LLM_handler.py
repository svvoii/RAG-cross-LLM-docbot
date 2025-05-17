import os
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_openai(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[ {"role": "user", "content": prompt} ],
            temperature=0.5,
            max_tokens=150,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "Error querying OpenAI API. Please check your API key and model."


def query_huggingface(prompt, model="gpt2"):
    generator = pipeline("text-generation", model=model)
    response = generator(prompt, max_new_tokens=150, num_return_sequences=1)
    return response[0]['generated_text']
