\documentclass[letterpaper, 11pt]{article}
\usepackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
\usepackage{times}
\usepackage[colorlinks=true, linkcolor=black, urlcolor=blue, citecolor=black]{hyperref}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{parskip}
\usepackage{multicol}

\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}

\titleformat{\section}{\normalsize\bfseries\uppercase}{}{0em}{}[{\titlerule[0.4pt]}]
\titlespacing{\section}{0pt}{10pt}{5pt}

\begin{document}

% -------------------------------------------------------
%  HEADER
% -------------------------------------------------------
{\Large\bfseries Evaluating and Detecting Hallucinations Across Large Language Models}

\smallskip
{\normalsize Project Proposal \quad|\quad [Your Name] \quad|\quad [Course Name]}

\vspace{6pt}
\hrule
\vspace{10pt}

% -------------------------------------------------------
\section{Problem Description}

Large Language Models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, legal research, and education. Despite their fluency, they frequently generate factually incorrect or unsupported content --- a phenomenon known as \textbf{hallucination}. Automated hallucination detection is a critical open problem in AI reliability and safety.

This project frames hallucination detection as a \textbf{binary classification task}: given an LLM-generated response to a factual question, predict whether it contains a hallucination. A secondary objective is to \textbf{compare hallucination rates across three LLMs} --- Gemini Flash, Llama 3.3 70B (via Groq), and Mistral 7B (via HuggingFace) --- surfacing behavioral differences across model families.

% -------------------------------------------------------
\section{Dataset}

\textbf{Primary --- HaluEval (Zhang et al., 2023):} A benchmark of $\sim$10,000 question-answer pairs with binary hallucination labels, covering general knowledge, summarization, and dialogue domains. Both ground truth and hallucinated answers are provided, enabling supervised classification.

\textbf{Supplementary --- TruthfulQA (Lin et al., 2022):} 817 questions designed to elicit hallucinations from LLMs. Each of the three target models will be queried via their respective free APIs; responses will be scored against ground truth using NLI entailment scores and BERTScore to produce a cross-model evaluation dataset.

\textbf{Preprocessing} will include text cleaning, embedding generation (\texttt{sentence-transformers}), NLI entailment score computation (DeBERTa-NLI via HuggingFace), and feature engineering covering response length, confidence markers, and semantic similarity to ground truth. Class imbalance will be addressed using SMOTE or inverse class weighting.

% -------------------------------------------------------
\section{Literature Review}

\begin{itemize}[leftmargin=*, itemsep=3pt, topsep=3pt]
    \item \textbf{HaluEval (Zhang et al., 2023):} Introduced the HaluEval benchmark and demonstrated that even GPT-4 hallucinates significantly on knowledge-intensive tasks.
    \item \textbf{TruthfulQA (Lin et al., 2022):} Showed that larger models are not necessarily more truthful; GPT-3 achieved only $\sim$58\% truthfulness on adversarially designed questions.
    \item \textbf{SelfCheckGPT (Manakul et al., 2023):} A sampling-based detection approach that checks consistency across multiple LLM outputs without requiring external knowledge, at the cost of multiple API calls per query.
    \item \textbf{FActScoring (Min et al., 2023):} Decomposes LLM responses into atomic claims and verifies each independently against a reference knowledge base.
\end{itemize}

This project contributes a systematic \textbf{multi-classifier benchmarking study} across three diverse LLM families, with emphasis on feature importance and cross-model failure mode analysis --- an angle underexplored in the above works. Results will be discussed in the context of prior literature rather than reproduced implementations.

% -------------------------------------------------------
\section{Algorithms}

Six classifiers will be trained on features extracted from LLM responses:

\begin{enumerate}[leftmargin=*, itemsep=2pt, topsep=3pt]
    \item \textbf{Logistic Regression} --- interpretable baseline well-suited for NLI score and embedding features
    \item \textbf{Support Vector Machine (SVM)} --- effective in high-dimensional text feature spaces
    \item \textbf{Random Forest} --- captures non-linear feature interactions; provides native feature importance
    \item \textbf{Gradient Boosting (XGBoost / LightGBM)} --- expected to be the strongest performing tabular classifier
    \item \textbf{DeBERTa-NLI} --- transformer-based end-to-end classifier via HuggingFace inference API (CPU-compatible)
    \item \textbf{k-Nearest Neighbors (KNN)} --- embedding-space similarity baseline
\end{enumerate}

% -------------------------------------------------------
\section{Evaluation}

Models will be evaluated using \textbf{macro F1, AUC-ROC, Precision, and Recall}, with F1 prioritized due to expected class imbalance. Validation will use stratified 5-fold cross-validation with a held-out test set for final evaluation. Hyperparameter tuning will be performed via \texttt{GridSearchCV} for Logistic Regression and SVM, and \texttt{RandomizedSearchCV} for XGBoost. Results will be contextualized against reported findings in Zhang et al.\ (2023) and Lin et al.\ (2022).

% -------------------------------------------------------
\section{Expected Results}

Gradient Boosting and DeBERTa-NLI are expected to achieve $>$75\% macro F1, outperforming classical baselines. Cross-model analysis is expected to reveal that smaller open-source models hallucinate more frequently than larger proprietary ones. NLI entailment score and semantic similarity to ground truth are anticipated to be the most predictive features. Should TruthfulQA-based generation yield insufficient data, HaluEval alone fully supports the classification pipeline, with cross-model comparison as a secondary contribution. The pipeline is designed to be extensible to additional models (e.g., GPT-4o, Claude Sonnet) as future work.

% -------------------------------------------------------
\section{Libraries and Tools}

\begin{table}[h!]
\centering
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}p{5.2cm}p{8.8cm}@{}}
\toprule
\textbf{Library / Tool} & \textbf{Purpose} \\
\midrule
\texttt{scikit-learn} & Classifiers, cross-validation, GridSearchCV \\
\texttt{xgboost} / \texttt{lightgbm} & Gradient boosting models \\
\texttt{transformers} (HuggingFace) & DeBERTa-NLI inference and classification \\
\texttt{sentence-transformers} & Semantic embedding generation \\
\texttt{pandas} / \texttt{numpy} & Data manipulation and feature engineering \\
\texttt{matplotlib} / \texttt{seaborn} & ROC curves, confusion matrices, feature importance plots \\
\texttt{google-generativeai} & Gemini Flash API (free tier, no credit card required) \\
\texttt{groq} & Llama 3.3 70B API (free tier) \\
\texttt{huggingface\_hub} & Mistral 7B free serverless inference endpoint \\
\texttt{imbalanced-learn} & SMOTE for handling class imbalance \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Tools to learn:} The HuggingFace \texttt{transformers} NLI inference pipeline and the Groq Python SDK are the two tools least familiar to the author. Both have well-documented quickstart guides that will be completed during week one of the project.

\end{document}