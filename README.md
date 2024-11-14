# Task1 - Retrieval (Search) System Evaluation

This project evaluates three embedding models—Universal Sentence Encoder (USE), BERT, and Cohere Embed v3—in a retrieval-based search system. The goal is to determine which model performs best for retrieving relevant documents based on query embeddings. Evaluation metrics include Precision@2, Recall@2, and Mean Reciprocal Rank (MRR).

## Project Structure

- **Dataset**: Synthetic dataset of queries and associated documents created for evaluating model performance.
- **Models**: 
  - **USE** (Universal Sentence Encoder)
  - **BERT** (Transformer-based model)
  - **Cohere Embed v3** (Proprietary embedding model)
- **Metrics**:
  - **Precision@2**: Measures the proportion of relevant documents in the top-2 results.
  - **Recall@2**: Assesses the retrieval of relevant documents within the top-2.
  - **MRR**: Focuses on ranking, evaluating the rank of the first relevant document in the result list.

## Key Findings

- **BERT**: Achieves the highest MRR score of 1.0, making it ideal for ensuring relevant documents appear at the top.
- **Cohere**: Provides balanced, high scores for both precision and recall, demonstrating reliable overall performance.
- **USE**: Underperforms across all metrics, suggesting limited effectiveness for this retrieval task.

## Usage

1. Clone the repository and navigate to the project directory.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the evaluation pipeline with `python Task1_Retrieval (Search) System Evaluation.py` to see model performance on the synthetic dataset.

## Visualization

The project includes a bar chart comparing model performance based on Precision@2, Recall@2, and MRR metrics.
![Bar_Plot](image.png)
