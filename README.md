# Detecting Hallucinations in Large Language Model Outputs
### A Feature-Based Classification Approach
**EECE 5644: Introduction to Machine Learning | Mohit Sharma**

---

## Overview

This project frames LLM hallucination detection as a binary classification task. Given a generated text response, we predict whether it contains a hallucination (factually incorrect or unsupported content) using 8 interpretable NLP features. We train and evaluate 6 classical ML classifiers on the HaluEval summarization benchmark, achieving a best F1 of **0.728** with Gradient Boosting.

A key finding of this project is the identification of a **length bias** in the HaluEval QA subset — where response length alone achieves 94% F1, rendering the task trivially easy. We document this investigation and switch to the summarization subset for a more rigorous evaluation.

---

## Repository Structure

```
DetectingHallucinationsInLLMOutputs/
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb                      # Load HaluEval + TruthfulQA
│   ├── 02_lllm_api_queries.ipynb                   # Query 3 LLMs on TruthfulQA
│   ├── 03_feature_engineering.ipynb                # Features for TruthfulQA responses
│   ├── 04_halueval_features.ipynb                  # Features for HaluEval QA (length bias investigation)
│   ├── 05_Modeling_and_Evaluation.ipynb            # Modeling on HaluEval QA (documents length bias)
│   ├── 06_halueval_summarization_features.ipynb    # Features for HaluEval summarization
│   └── 07_modelling_summarization.ipynb            # Final modeling on summarization subset
│
├── data/
│   ├── raw/                                        # Empty — datasets downloaded via HuggingFace API
│   └── processed/
│       ├── halueval_classification.csv             # HaluEval QA reshaped (20,000 rows)
│       ├── halueval_features.csv                   # HaluEval QA features (5,000 rows)
│       ├── halueval_sum_features.csv               # HaluEval summarization features (5,000 rows)
│       ├── truthfulqa_clean.csv                    # TruthfulQA cleaned (817 rows)
│       ├── llm_responses.csv                       # LLM responses to TruthfulQA (817 rows)
│       ├── llm_responses_features.csv              # Features for LLM responses (817 rows)
│       ├── llm_responses_checkpoint.csv            # API query checkpoint
│       └── llm_requery_checkpoint.csv              # Re-query checkpoint for failed rows
│
├── outputs/
│   ├── figures/
│   │   ├── halueval_eda.png
│   │   ├── truthfulqa_eda.png
│   │   ├── halueval_feature_distributions.png
│   │   ├── halueval_sum_feature_distributions.png
│   │   ├── baseline_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   ├── sum_baseline_comparison.png
│   │   ├── sum_confusion_matrices.png
│   │   ├── sum_roc_curves.png
│   │   ├── sum_feature_importance.png
│   │   └── sum_cross_model_analysis.png
│   ├── models/                                     # Saved model files
│   ├── sum_tuned_model_results.csv
│   ├── sum_tuned_vs_baseline.csv
│   └── sum_cross_model_hallucination_rates.csv
│
├── report/                                         # LaTeX report files
├── src/                                            # Utility functions
├── EECE5644ProjectProposal.pdf                     # Original project proposal
├── EECE5644ProjectMohitSharma-DetectingHallucinationsinLLMs.pptx
├── requirements.txt
└── README.md
```

---

## Key Results

| Model | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| **Gradient Boosting** | **0.728** | 0.746 | 0.710 | 0.801 |
| Random Forest | 0.726 | 0.738 | 0.714 | 0.794 |
| Logistic Regression | 0.723 | 0.752 | 0.696 | 0.805 |
| SVM | 0.722 | 0.759 | 0.688 | 0.806 |
| KNN | 0.721 | 0.723 | 0.720 | 0.784 |
| Decision Tree | 0.691 | 0.708 | 0.674 | 0.758 |

---

## Datasets

| Dataset | Source | Used For |
|---|---|---|
| HaluEval (summarization) | [HuggingFace](https://huggingface.co/datasets/pminervini/HaluEval) | Training & evaluating classifiers |
| HaluEval (QA) | Same | Length bias investigation |
| TruthfulQA | [HuggingFace](https://huggingface.co/datasets/truthful_qa) | LLM response generation |

---

## Features

Eight NLP features computed per response against the source document:

| Feature | Description |
|---|---|
| `response_len` | Word count of the response |
| `hedge_count` | Count of uncertainty markers (might, possibly, etc.) |
| `rouge1` | Unigram overlap with source (ROUGE-1 F1) |
| `rougeL` | Longest common subsequence overlap (ROUGE-L F1) |
| `bleu` | N-gram precision (smoothed sentence BLEU) |
| `sem_sim` | Cosine similarity of sentence embeddings (all-MiniLM-L6-v2) |
| `bertscore_f1` | Token-level semantic similarity (distilbert-base-uncased) |
| `nli_score` | NLI entailment score (cross-encoder/nli-deberta-v3-small) |

---

## LLMs Queried

| Model | Provider | API |
|---|---|---|
| Mistral Small | Mistral AI | console.mistral.ai (free tier) |
| Llama 3.3 70B | Meta via Groq | console.groq.com (free tier) |
| Llama 3.1 8B | Meta via SambaNova/HF | huggingface.co (free tier) |

---

## Setup & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hallucination-detection.git
cd hallucination-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy datasets sentence-transformers transformers
pip install bert-score rouge-score nltk xgboost scikit-learn
pip install google-generativeai groq huggingface_hub mistralai
pip install python-dotenv tqdm matplotlib seaborn
```

### 3. Set up API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
HF_API_TOKEN=your_huggingface_token
MISTRAL_API_KEY=your_mistral_key
```

### 4. Run notebooks in order
```
01 → 02 → 06 → 07
```
Note: Notebooks 04 and 05 document the length bias investigation and should be read for context but are not required to reproduce final results.

---

## Important Notes

- **No GPU required** — all feature extraction and modeling runs on CPU
- **Notebook 03b / 04** are intentionally retained to document the investigative process of discovering the HaluEval QA length bias
- API queries in Notebook 02 take approximately 3–4 sessions of 15–20 minutes each due to free tier rate limits
- BERTScore and NLI computation in Notebook 03c takes approximately 45–60 minutes on CPU

---

## Citation

```bibtex
@inproceedings{zhang2023halueval,
  title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  author={Zhang, Yue and Li, Yafu and Cui, Leyang and Cai, Deng and Liu, Lemao and Fu, Tingchen and Huang, Xinting and Zhao, Enbo and Zhang, Yu and Chen, Yulong and Wang, Longyue and Luu, Anh Tuan and Bi, Wei and Shi, Freda and Shi, Shuming},
  booktitle={Proceedings of EMNLP 2023},
  year={2023}
}

@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={Proceedings of ACL 2022},
  year={2022}
}
```

---

## Academic Integrity Disclosure

This project was completed with assistance from Claude (Anthropic) for brainstorming, debugging, and writing guidance. All code, analysis, and written content were reviewed, understood, and validated by the author. AI assistance was used in accordance with course policies.
