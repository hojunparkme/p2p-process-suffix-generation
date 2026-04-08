# Zero-Shot Process Suffix Generation via Transition Probability-Guided LLM Reasoning

**Ho Jun Park** | Industrial and Information Systems Engineering, Soongsil University

---

![Framework Architecture](assets/framework.svg)

---

## Overview

Supervised approaches to predictive process monitoring are structurally limited to the prefix sequence — they predict the most statistically likely continuation from historical patterns, with no mechanism to account for the actual business context of a running case. Two cases with identical prefix sequences may require entirely different resolution paths depending on the underlying business problem, yet a supervised model receiving identical inputs produces identical predictions.

**Key insight:** Process participants already possess contextual knowledge about their current situation. Our framework captures this knowledge through a natural language query and combines it with transition probability-based candidate filtering to guide an LLM in generating the complete activity suffix — without any model training or expert annotation.

Evaluated on the BPI Challenge 2019 dataset (251,734 cases, 42 activities), our method outperforms supervised baselines including Tax LSTM and SuTraN across all six event categories on both Damerau-Levenshtein similarity and F1.

---

## Results

| Method | DL Similarity | F1 |
|---|---|---|
| SuTraN (2024) | 0.403 | 0.529 |
| Tax LSTM (2017) | 0.507 | 0.634 |
| LLM-only (ablation) | 0.652 | 0.786 |
| **Ours** | **0.714** | **0.839** |

| Event Category | SuTraN | Tax LSTM | LLM-only | Ours |
|---|---|---|---|---|
| Cancel Invoice Receipt | 0.391 / 0.528 | 0.606 / 0.669 | 0.531 / 0.717 | **0.606 / 0.762** |
| Change Delivery Indicator | 0.368 / 0.461 | 0.461 / 0.553 | 0.530 / 0.661 | **0.604 / 0.761** |
| Change Price | 0.425 / 0.604 | 0.551 / 0.699 | 0.637 / 0.752 | **0.729 / 0.838** |
| Change Quantity | 0.384 / 0.497 | 0.484 / 0.617 | 0.623 / 0.776 | **0.666 / 0.802** |
| Remove Payment Block | 0.421 / 0.541 | 0.590 / 0.696 | 0.817 / 0.906 | **0.854 / 0.948** |
| Vendor creates debit memo | 0.465 / 0.598 | 0.330 / 0.545 | 0.696 / 0.837 | **0.813 / 0.914** |

*(DL Similarity / F1)*

---

## Setup

1. Clone the repository

        git clone https://github.com/hojunparkme/p2p-process-suffix-generation.git

2. Install dependencies

        pip install anthropic python-dotenv numpy

3. Create `.env` file

        ANTHROPIC_API_KEY=your_api_key_here

4. Download [BPI Challenge 2019](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) and place `BPI_Challenge_2019.xes` in `data/`

---

## Usage

Run our framework:

    cd src/experiment && python claude_experiment_final2.py

Run Tax LSTM baseline:

    cd src/baselines && python tax_lstm_torch.py

SuTraN evaluation (requires [official repo](https://github.com/BrechtWts/SuffixTransformerNetwork)):

    cd src/baselines && python sutran_qa_eval.py

---

## Citation

    @article{park2025p2p,
      title={Zero-Shot Process Suffix Generation via Transition Probability-Guided LLM Reasoning in Purchase-to-Pay Processes},
      author={Park, Ho Jun and Lee, Younsoo and Kang, Changmuk},
      year={2025}
    }

---

## Related Work
- Tax et al. (2017) - Predictive Business Process Monitoring with LSTM Neural Networks, CAiSE
- Wuyts et al. (2024) - SuTraN: Encoder-Decoder Transformer for Suffix Prediction, ICPM
- Bukhsh et al. (2021) - ProcessTransformer: Predictive Business Process Monitoring with Transformer
- Rama-Maneiro et al. (2021) - Deep Learning for Predictive Business Process Monitoring: Review and Benchmark, IEEE TSC
- Pasquadibisceglie et al. (2024) - LUPIN: LLM Approach for Activity Suffix Prediction, ICPM
- Oved et al. (2025) - SNAP: Semantic Stories for Next Activity Prediction, AAAI
- Casciani et al. (2026) - Enhancing Next Activity Prediction with RAG, Information Systems
- Park et al. (in press) - Structure-Preserving Process Case Embeddings via Directed Graph Convolutional Networks, ICIC Express Letters
- van Dongen (2019) - BPI Challenge 2019 Dataset, 4TU
