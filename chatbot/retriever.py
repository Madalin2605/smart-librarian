import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


openai_api_key = os.getenv("OPENAI_API_KEY")

chroma_client = chromadb.PersistentClient(path="db/chroma_db")
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="book_summaries",
    embedding_function=embedding_function
)


def parse_book_summaries(file_path: str):
    """
    Parse a plaintext file of book summaries into (documents, metadatas, ids).

    Expected file format:
        ## Title: The Book Title
        This is the first line of the summary.
        This is the second line of the summary.
        ... (more lines)

        ## Title: Another Title
        Another summary...

    The function splits on '## Title:' and assumes:
      - The first line after '## Title:' is the book title.
      - The remaining lines in that chunk form the summary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = content.split("## Title:")
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        if chunk.strip() == "":
            continue
        lines = chunk.strip().split("\n")
        title = lines[0].strip()
        summary = " ".join(lines[1:]).strip()
        documents.append(summary)
        metadatas.append({"title": title})
        ids.append(f"book_{i}")

    return documents, metadatas, ids


def populate_chroma():
    """
    Load book summaries from disk and insert them into the Chroma collection
    — but only if the collection is currently empty.
    """
    file_path = "data/book_summaries.txt"
    documents, metadatas, ids = parse_book_summaries(file_path)
    existing = collection.count()

    if existing == 0:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Loaded {len(documents)} summaries into ChromaDB.")
    else:
        print(f"ChromaDB already populated with {existing} entries.")


def search_books(query: str, n_results: int = 2):
    """
    Run a semantic search over the 'book_summaries' collection.

    Args:
        query: Natural language search string (any language).
        n_results: How many top matches to return.

    Returns:
        The Chroma query result dict, including documents, metadatas, distances, and ids.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    
    return results


if __name__ == "__main__":

     # Ensure the collection is populated before searching.
    populate_chroma()

    # Example query in Romanian asking for a book recommendation about freedom and social control.
    test_query = "Vreau o carte despre fotbal."
    results = search_books(test_query)

    # Print the titles of the matched books for quick inspection.
    for match in results['metadatas'][0]:
        print(f"Title: {match['title']}")
