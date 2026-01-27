import cohere

class CohereReranker:
    def __init__(self, client: cohere.Client, model: str = "rerank-v4.0-fast"):
        self.co = client
        self.model = model

    def rank_text(self, query: str, documents: List[str], top_n: int = 3)
        result = self.co(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n
        )

        return result.results