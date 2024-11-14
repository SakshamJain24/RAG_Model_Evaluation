import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from transformers import AutoTokenizer, AutoModel
import torch
import cohere
import numpy as np
import matplotlib.pyplot as plt


class EmbeddingEvaluator:
    def __init__(self, cohere_api_key):
        self.cohere_client = cohere.Client(cohere_api_key)

        print("Loading USE model...")
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        print("Loading BERT model...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')

    def get_use_embeddings(self, texts):
        embeddings = self.use_model(texts)
        return embeddings.numpy()

    def get_bert_embeddings(self, texts):
        encoded_input = self.bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.bert_model(**encoded_input)
        sentence_embeddings = model_output.last_hidden_state[:, 0, :]
        return sentence_embeddings.numpy()

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
    # Calculate average metrics per model
    comparative_analysis = results_df.groupby("Model")[["Precision@2", "Recall@2", "MRR"]].mean().round(3)
    comparative_analysis = comparative_analysis.reset_index()

    # Create visualization
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
    data = {
        "Query Text": [
            "Where was Elon Musk born?",
            "What was Musk's childhood like?",
            "How did Musk get interested in technology?"
        ],
        "Document 1": [
            "Elon Musk was born in 1971 in Pretoria, South Africa, the oldest of three children. His father was a South African engineer, and his mother was a Canadian model and nutritionist.",
            "Musk had a difficult childhood, being bullied at school, where gangs chased and beat him. At home, his relationship with his father was strained.",
            "At the age of 10, Musk started programming with a Commodore VIC-20 and created his first video game, Blastar. He sold the game's code for $500."
        ],
        "Document 2": [
            "Musk was born in Pretoria, South Africa, in 1971, to a South African engineer father and a Canadian model mother.",
            "Musk described his childhood as challenging, being targeted by school gangs and dealing with a tumultuous home life after his parents' divorce.",
            "Musk became interested in technology at the age of 10 when he learned programming on a Commodore VIC-20, eventually selling a game he created."
        ],
        "Document 3": [
            "Elon Musk was born in 1971 in Pretoria, South Africa. His parents divorced when he was young, and he lived with his father.",
            "Growing up was tough for Musk, as he was bullied at school and had a strained relationship with his father, whom he later described as a \" terrible human being.\"",
            "By 10, Musk had already taught himself programming and created a game called Blastar on a Commodore VIC-20, which he sold for $500."
        ]
    }

    COHERE_API_KEY = "zrKybib5xg7zuRUukXm97KQVOrQitTLcin3Y7kW2"

    # Run evaluation
    results_df = evaluate_embeddings(data, COHERE_API_KEY)

    # Generate comparative analysis and visualization
    comparative_analysis, plot = create_comparative_analysis(results_df)

    # Print final comparative analysis table
    print("\nFinal Comparative Analysis:")
    print(comparative_analysis.to_string(index=False))

    # Display detailed results if needed
    print("\nDetailed Results per Query:")
    detailed_results = results_df[["Query", "Model", "Precision@2", "Recall@2", "MRR"]]
    print(detailed_results.to_string())

    # Show visualization
    plot.show()