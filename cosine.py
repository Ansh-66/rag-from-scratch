import numpy as np

def cosine_similarity(a, b):
    a_array = np.array(a)
    b_array = np.array(b)

    dot  = np.dot(a, b)
    norm_a = np.linalg.norm(a_array)
    norm_b = np.linalg.norm(b_array)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("The magnitude of either vectors cannot be Zero")
    return dot / (norm_a * norm_b)

# print(cosine_similarity([1,2,3], [1,2,3]))
# print(cosine_similarity([1,0], [0,1]))