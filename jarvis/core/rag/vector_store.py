

class VectorStore:
    def __init__(self, embedder: CohereEmbedder):
        self.embedder = embedder

    def add(self, text: str, metadata: dict = None) -> str:
        pass

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        pass

    def delete(self, id: str) -> bool:
        pass

    def load_from_file(self, filepath: str):
        pass

    def save_to_file(self, filepath: str):
        pass