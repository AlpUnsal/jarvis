import cohere

class CohereEmbedder:
    def __init__(self, co: cohere.Client, model: str = "embed-v4.0"):
        self.co = client
        self.model = model
    
    def embed_text(self, texts: List[str], input_type: str = "search_document") -> List[float]:
        result = self.co.embed(
            model=self.model,
            input_type=input_type,
            texts=texts,
            embedding_types=["float"]
        )

        return result.embeddings

    def embed_image(self, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
        pass