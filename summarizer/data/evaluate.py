# evaluate.py
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from preprocess import clean_text
from predict import summarize_text

# BLEU Score (translation metric)

def compute_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)


# ROUGE Score (summarization metric)

def compute_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


# Evaluate dataset samples

def evaluate_model(articles, references, num_samples=100):
    bleu_scores = []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    for i in range(min(num_samples, len(articles))):
        ref = clean_text(references[i])
        cand = summarize_text(articles[i])
        
        bleu_scores.append(compute_bleu(ref, cand))
        rouge = compute_rouge(ref, cand)
        rouge1_scores.append(rouge['rouge1'].fmeasure)
        rouge2_scores.append(rouge['rouge2'].fmeasure)
        rougeL_scores.append(rouge['rougeL'].fmeasure)
    
    print("Average BLEU:", np.mean(bleu_scores))
    print("Average ROUGE-1:", np.mean(rouge1_scores))
    print("Average ROUGE-2:", np.mean(rouge2_scores))
    print("Average ROUGE-L:", np.mean(rougeL_scores))

if __name__ == "__main__":
    # Example usage (replace with dataset)
    articles = [
        "Tim Cook is the CEO of Apple since 2011. He succeeded Steve Jobs.",
        "Elon Musk founded SpaceX and co-founded Tesla."
    ]
    references = [
        "Tim Cook became Apple CEO in 2011 after Steve Jobs.",
        "Elon Musk founded SpaceX and Tesla."
    ]
    
    evaluate_model(articles, references, num_samples=2)
