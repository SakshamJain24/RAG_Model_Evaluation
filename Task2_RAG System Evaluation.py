import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import pandas as pd


class TextGeneratorEvaluator:
    def __init__(self):
        # Initialize GPT-2 Model
        print("Loading GPT-2 model...")
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

        # Initialize BART Model for text generation
        print("Loading BART model...")
        self.bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

        # Initialize T5 Model (Google T5 as an example LLM)
        print("Loading T5 model...")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def generate_gpt2(self, prompt):
        input_ids = self.gpt2_tokenizer(prompt, return_tensors="pt").input_ids
        output = self.gpt2_model.generate(input_ids, max_length=100)
        return self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_bart(self, prompt):
        input_ids = self.bart_tokenizer(prompt, return_tensors="pt").input_ids
        output = self.bart_model.generate(input_ids, max_length=50)
        return self.bart_tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_t5(self, prompt):
        input_ids = self.t5_tokenizer(prompt, return_tensors="pt").input_ids
        output = self.t5_model.generate(input_ids, max_length=50)
        return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)

    def evaluate_metrics(self, generated_text, reference_text):
        # ROUGE
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = rouge.score(generated_text, reference_text)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure

        # BLEU
        reference = reference_text.split()
        candidate = generated_text.split()
        bleu_score = sentence_bleu([reference], candidate)

        # BERTScore
        P, R, F1 = bert_score([generated_text], [reference_text], lang="en")
        bert_f1 = F1.mean().item()

        return {
            "ROUGE-1": rouge1,
            "ROUGE-L": rougeL,
            "BLEU": bleu_score,
            "BERTScore": bert_f1
        }

    def process_queries(self, data, reference_texts):
        results = []

        for i, query in enumerate(data["Query Text"]):
            retrieved_texts = [
                data["Document 1"][i],
                data["Document 2"][i]
            ]
            prompt = f"{query}\n{retrieved_texts[0]}\n{retrieved_texts[1]}\nGenerate a response based on the above information."

            # Generate outputs from three models
            bart_response = self.generate_bart(prompt)
            gpt2_response = self.generate_gpt2(prompt)
            t5_response = self.generate_t5(prompt)

            # Evaluate generated outputs against the reference text
            for model_name, generated_text in zip(
                    ["BART", "GPT-2", "T5"],  # Include all models
                    [bart_response, gpt2_response, t5_response]
            ):
                metrics = self.evaluate_metrics(generated_text, reference_texts[i])
                results.append({
                    "Query": query,
                    "Model": model_name,
                    "Generated Text": generated_text,
                    "ROUGE-1": metrics["ROUGE-1"],
                    "ROUGE-L": metrics["ROUGE-L"],
                    "BLEU": metrics["BLEU"],
                    "BERTScore": metrics["BERTScore"]
                })

        return pd.DataFrame(results)


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
        ]
    }

    reference_texts = [
        "Elon Musk was born in Pretoria, South Africa, in 1971.",
        "Musk's childhood was tough; he faced bullying and had a strained family life.",
        "Musk's interest in technology began at 10 when he learned programming and created his first game."
    ]

    evaluator = TextGeneratorEvaluator()
    results_df = evaluator.process_queries(data, reference_texts)

    # Display the results
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))
