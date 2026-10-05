def store(chunks, vectors):
    storage_list = []
    if len(chunks) != len(vectors):
        raise ValueError("The length of Chunks list and Vectors list should be same")
    for chunk, vector in zip(chunks, vectors):
        st = {"text": chunk, "vector": vector}
        storage_list.append(st)
    return storage_list

# result = store(["a", "b", "c"], [[1.0], [2.0], [3.0]])
# print(len(result))
# print(result[0])

# result2 = store(["a", "b"], [[1.0]])
# print(result2)
if __name__ == "__main__":
    print(store([], []))