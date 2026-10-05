import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def generate_answer(question, chunks, client):
    context = "\n\n".join(chunks)
    prompt = f"""
You must answer the user's question using **only** the facts provided in the context below. Do not use any outside knowledge, assumptions, or extrapolations.
You are permitted to synthesize and combine information across multiple passages within the provided context. Note that combining facts explicitly stated in the context does not constitute using outside knowledge.
If the answer cannot be found or derived from the context, you must state: "I cannot answer this based on the provided information." Do not attempt to make up an answer or speculate.
---
Context:
--------------------
{context}
--------------------
---
Question:
{question}
"""
    response = client.models.generate_content(model = "gemini-3.5-flash-lite", contents = prompt)
    return response.text

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key = api_key)

    print(generate_answer("How tall is the Eiffel Tower?", ["The Eiffel Tower is 330 metres tall.", "It was completed in 1889."], client))
    print(generate_answer("What is the capital of France?", ["Sourdough bread requires a starter of flour and water.", "Let the dough rise for several hours."], client))