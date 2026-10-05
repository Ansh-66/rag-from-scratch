from chunker import chunker
from embed_chunks import embed_chunks
from store import store
from retrieve import embed_query
from retrieve import retrieve_chunks
from generate import generate_answer
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from eval import evaluate_recall, labelled_set
from rerank import rerank

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    with open("source.txt") as f:
        source = f.read()
    
    client = genai.Client(api_key = api_key)
    chunks = chunker(source, 600, 60)
    vectors = embed_chunks(chunks, client)
    storage = store(chunks, vectors)

    question = input("Enter your question: ")
    query_vector = embed_query(question, client)
    candidates = retrieve_chunks(query_vector, storage, 5)
    reranked = rerank(question, candidates, client, 3)
    texts = [item["text"] for item in reranked]
    answer = generate_answer(question, texts, client)

    print(answer)
    result1 = evaluate_recall(labelled_set, storage, client, 1)
    result2 = evaluate_recall(labelled_set, storage, client, 2)
    result3 = evaluate_recall(labelled_set, storage, client, 3)
    print(result1["recall"])
    print(result2["recall"])
    print(result3["recall"])
    # print(result["misses"])


if __name__ == "__main__":
    main()
