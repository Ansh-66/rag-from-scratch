def chunker(doc, chunk_size, overlap):
    chunks = []
    if overlap >= chunk_size:
        raise ValueError("Overlap must not be greater than or equal to chunk_size")
    if len(doc) == 0:
        return []

    stride = chunk_size - overlap
    start = 0

    while True:
        chunk = doc[start : start + chunk_size]
        chunks.append(chunk)
        if start+chunk_size>=len(doc):
            break
        start = start + stride

    return chunks


if __name__ == "__main__":
    doc = "a" * 170
    result = chunker(doc, 100, 20)
    print(len(result))
    for c in result:
        print(len(c))
    print(chunker("", 100, 20))

# print(chunker("a" * 170, 100, 100))
# print(chunker("a" * 50, 100, 20))
# print(chunker("a" * 200, 100, 20))