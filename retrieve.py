import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from cosine import cosine_similarity


def embed_query(query, client):
    try:
        response = client.models.embed_content(model = 'gemini-embedding-001', contents = query, config = types.EmbedContentConfig(task_type='RETRIEVAL_QUERY'))
        query_vector = response.embeddings[0].values
        return query_vector
    except Exception as e:
        print("Error: ", e)
        raise ValueError("Unable to embed query")

def retrieve_chunks(query_vector, storage, k):

    results = []
    for item in storage:    
        score = cosine_similarity(query_vector, item["vector"])
        results.append({"chunk": item, "score": score})
    
    results.sort(key=lambda r: r["score"], reverse=True)
    return [w["chunk"] for  w in results[:k]]


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key = api_key)
    storage = [
        {"text": "cat",      "vector": [1.0, 0.0]},
        {"text": "kitten",   "vector": [0.9, 0.1]},
        {"text": "airplane", "vector": [0.0, 1.0]},
    ]


    print(retrieve_chunks([0.0, 1.0], storage, 2))