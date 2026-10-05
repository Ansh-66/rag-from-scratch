import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


def embed_chunks(chunks, client):
    vectors = []
    for chunk in chunks:
        response = client.models.embed_content(model = 'gemini-embedding-001', contents = chunk, config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT'))
        vectors.append(response.embeddings[0].values)
    return vectors

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key = api_key)
    # test_chunks = ["the sky is blue", "grass is green", "sun is hot"]
    test_chunks = ["airplane", "cats", "dogs"]
    result = embed_chunks(test_chunks, client)
    print(result)
    print(len(result))
    print(len(result[0]))