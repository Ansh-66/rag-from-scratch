import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import re

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key = api_key)

candidates = [
    {"text": "octopuses live three to five years"},
    {"text": "coffee water should be 90 to 96 degrees"},
    {"text": "Gutenberg introduced printing around 1440"},
]

k = 2
query = "How long do Octopuses live?"

lines = []
for number, candidate in enumerate(candidates, start = 1):
    lines.append(f"{number}. {candidate['text']}")
block = "\n".join(lines)
print(block)

prompt = f"""
Query:
{query}
Numbered Candidates:
{block}
Instruction:
Rank the candidates above based on their relevance to the query. Return the candidate numbers in relevance order, most relevant first, numbers only, comma-separated, nothing else.
"""

response = client.models.generate_content(model = 'gemini-3.5-flash-lite', contents = prompt)
reply = response.text

extracted = [int(n) for n in re.findall(r'\d+', reply)]
if sorted(extracted ) == list(range(1, len(candidates)+1)):
    reordered = [candidates[x-1] for x in extracted]
    reordered = reordered[:k]
else:
    reordered = candidates[:k]


print(reordered)
    
