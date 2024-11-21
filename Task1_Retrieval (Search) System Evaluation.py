import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import cohere
import numpy as np
import matplotlib.pyplot as plt
from corpus_task1 import data
from secrets import COHERE_API_KEY


class EmbeddingEvaluator:
    def __init__(self, cohere_api_key):
        self.cohere_client = cohere.Client(cohere_api_key)

        print("Loading USE model...")
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        print("Loading SentenceTransformer BERT model...")
        self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_use_embeddings(self, texts):
        embeddings = self.use_model(texts)
        return embeddings.numpy()

    def get_bert_embeddings(self, texts):
        # Generating embeddings using SentenceTransformer
        embeddings = self.bert_model.encode(texts, convert_to_numpy=True)
        return embeddings

    def get_cohere_embeddings(self, texts):
        response = self.cohere_client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_query'
        )
        return np.array(response.embeddings)

    def calculate_similarities(self, query_embedding, document_embeddings):
        similarities = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)[0]
        return similarities

    def get_rankings(self, similarities):
        return np.argsort(-similarities)

    def calculate_metrics(self, retrieved_ranks, relevant_docs, k=2):
        top_k = retrieved_ranks[:k]
        relevant_in_k = sum(1 for doc in top_k if doc in relevant_docs)
        precision_k = relevant_in_k / k if k > 0 else 0
        recall_k = relevant_in_k / len(relevant_docs) if relevant_docs else 0

        for rank, doc in enumerate(retrieved_ranks, 1):
            if doc in relevant_docs:
                mrr = 1 / rank
                break
        else:
            mrr = 0

        return precision_k, recall_k, mrr


def evaluate_embeddings(data, cohere_api_key):
    evaluator = EmbeddingEvaluator(cohere_api_key)
    results = []

    for query_idx, query in enumerate(data["Query Text"]):
        print(f"\nProcessing Query {query_idx + 1}: {query}")
        documents = [data[f"Document {i + 1}"][query_idx] for i in range(3)]
        relevant_docs = {0, 1}

        for model_name in ["USE", "BERT", "Cohere"]:
            print(f"Processing with {model_name}...")

            if model_name == "USE":
                query_embedding = evaluator.get_use_embeddings([query])
                doc_embeddings = evaluator.get_use_embeddings(documents)
            elif model_name == "BERT":
                query_embedding = evaluator.get_bert_embeddings([query])
                doc_embeddings = evaluator.get_bert_embeddings(documents)
            else:
                query_embedding = evaluator.get_cohere_embeddings([query])
                doc_embeddings = evaluator.get_cohere_embeddings(documents)

            similarities = evaluator.calculate_similarities(query_embedding, doc_embeddings)
            rankings = evaluator.get_rankings(similarities)
            precision_k, recall_k, mrr = evaluator.calculate_metrics(rankings, relevant_docs, k=2)

            results.append({
                "Query": query,
                "Model": model_name,
                "Precision@2": precision_k,
                "Recall@2": recall_k,
                "MRR": mrr,
                "Rankings": rankings.tolist(),
                "Similarities": similarities.tolist()
            })

    return pd.DataFrame(results)


def create_comparative_analysis(results_df):
    # Calculating average metrics per model
    comparative_analysis = results_df.groupby("Model")[["Precision@2", "Recall@2", "MRR"]].mean().round(3)
    comparative_analysis = comparative_analysis.reset_index()

    # Creating visualization
    plt.figure(figsize=(10, 6))
    metrics = ["Precision@2", "Recall@2", "MRR"]
    x = np.arange(len(comparative_analysis["Model"]))
    width = 0.25

    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, comparative_analysis[metric], width, label=metric)

    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Comparative Analysis of Embedding Models")
    plt.xticks(x + width, comparative_analysis["Model"])
    plt.legend()
    plt.tight_layout()

    return comparative_analysis, plt


if __name__ == "__main__":

    # Running evaluation
    results_df = evaluate_embeddings(data, COHERE_API_KEY)

    # Generating comparative analysis and visualization
    comparative_analysis, plot = create_comparative_analysis(results_df)

    # Printing final comparative analysis table
    print("\nFinal Comparative Analysis:")
    print(comparative_analysis.to_string(index=False))

    # Displaying detailed results
    print("\nDetailed Results per Query:")
    detailed_results = results_df[["Query", "Model", "Precision@2", "Recall@2", "MRR"]]
    print(detailed_results.to_string())

    # Show visualization
    plot.show()
