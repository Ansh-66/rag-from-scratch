import os 
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieve import embed_query, retrieve_chunks
from rerank import rerank

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key = api_key)

labelled_set = [
  {"question": "What is the typical weight of an adult giant Pacific octopus?", "answer_phrase": "15 kilograms"},
  {"question": "What is the arm span of an adult giant Pacific octopus?", "answer_phrase": "4.3 metres"},
  {"question": "Where does the giant Pacific octopus live?", "answer_phrase": "northern Pacific Ocean"},
  {"question": "At what depths is the giant Pacific octopus most often found?", "answer_phrase": "less than 100 metres"},
  {"question": "What predators do the giant Pacific octopus use camouflage against?", "answer_phrase": "seals and large fish"},
  {"question": "How long do most giant Pacific octopuses live?", "answer_phrase": "three to five years"},
  {"question": "When does the male giant Pacific octopus die during reproduction?", "answer_phrase": "a few months after mating"},
  {"question": "What does the female giant Pacific octopus stop doing once she lays her eggs?", "answer_phrase": "stops eating"},
  {"question": "What grind size suits methods with long contact time like a French press?", "answer_phrase": "coarse grind"},
  {"question": "What grind size is needed for espresso?", "answer_phrase": "fine grind"},
  {"question": "What is the most common cause of bitter or weak coffee at home?", "answer_phrase": "wrong grind"},
  {"question": "What happens if water is above roughly 96 degrees Celsius when brewing coffee?", "answer_phrase": "scorches the grounds"},
  {"question": "What water temperature do most brewing guides recommend?", "answer_phrase": "90 and 96 degrees Celsius"},
  {"question": "What is a common starting point for the ratio of coffee to water?", "answer_phrase": "60 grams of coffee per litre of water"},
  {"question": "When did Johannes Gutenberg introduce movable-type printing to Europe?", "answer_phrase": "around 1440"},
  {"question": "What was Gutenberg's key innovation?", "answer_phrase": "casting large quantities of durable, reusable metal type"},
  {"question": "Which historical movements do historians often link the printing press to?", "answer_phrase": "Reformation and the Scientific Revolution"}
]

def evaluate_recall(labelled_set, storage, client, k):
    counter = 0
    misses = []
    for case in labelled_set:
        query_vector = embed_query(case["question"], client)
        candidates = retrieve_chunks(query_vector, storage, 5)
        reranked = rerank(case["question"], candidates, client, k)
        if any(case["answer_phrase"] in chunk["text"] for chunk in reranked):
            counter = counter + 1
        else:
            misses.append(case["question"])

        time.sleep(5)

    return {
        "recall": counter / len(labelled_set),
        "misses": misses
    }

