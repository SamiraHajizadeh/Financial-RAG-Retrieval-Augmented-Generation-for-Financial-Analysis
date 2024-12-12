# Financial RAG

## Overview

The GenAI Financial Agent is a conversational AI system designed to assist users with financial queries. It leverages the powerful **Qwen 2.5** language model, fine-tuned on financial datasets using **LoRA** and **4-bit quantization** for efficient resource utilization. 

To enhance the accuracy and context-awareness of the generated responses, the agent integrates **Retrieval-Augmented Generation (RAG)**. This approach allows the model to access and incorporate relevant information from external knowledge sources during the generation process. 

The agent supports two RAG implementations:

- **Naive RAG:** A straightforward retrieval-based approach where relevant information is retrieved and directly used in the generation process.
- **Agentic RAG:** A more sophisticated approach that enables the model to make informed decisions about which information to retrieve and how to best utilize it for the given task.

**Note:** The current implementation utilizes **Wealth Alpaca** and **Financial QA 10k** as the primary knowledge sources for RAG.

## Features

- **Fine-Tuned Qwen 2.5 Model:**
    - Optimized using **QLoRA** for parameter-efficient training, enabling efficient training and deployment.
    - Trained on a curated dataset of **Wealth Alpaca** to specialize in financial domain knowledge. 
- **Retrieval-Augmented Generation (RAG):**
    - **Naive RAG:** Implements a basic retrieval-based approach for generating responses.
    - **Agentic RAG:** Enables more sophisticated decision-making within the RAG framework for improved performance on complex financial tasks.
    - **Fine-Tuned RAG:** Leverages the fine-tuned Qwen 2.5 model with **Wealth Alpaca** and **Financial QA 10k** as knowledge sources for refined retrieval-based answer generation.
- **Interactive Inference:**
    - Enables real-time question-answering using the fine-tuned model, providing users with immediate responses to their financial inquiries. 

## Getting Started

### Evaluating RAG and Agentic RAG on a Financial Q&A Dataset Compared to Baseline Qwen-2.5 3B

You can import the code to your IDE using:

```bash
git clone https://github.com/SamiraHajizadeh/Financial_RAG.git
cd Financial_RAG
pip install -r requirements.txt
```

Afterwards, change Cell 4 in the notebook before running the code.

<img src="image.png" width="400" />

### Fine-tuning Qwen2.5 3B on Wealth Alpaca

Something

## Evaluation Results

### Performance Comparison
The system was evaluated across multiple metrics using the FinQA dataset:

| Metric                 | Baseline Model | Fine-Tuned Model | Naive RAG | Agentic RAG |
|------------------------|----------------|------------------|-----------|-------------|
| Cosine Similarity      | 0.74           | 0.70             | 0.86      | 0.79        |
| BLEU Score             | 0.005          | 0.00             | 0.42      | 0.24        |
| Answer Relevance       | 0.84           | -                | 0.95      | 0.95        |
| Faithfulness           | N/A            | N/A              | 0.89      | N/A         |
| Context Precision      | N/A            | N/A              | 0.92      | N/A         |

Key Findings:
- Naive RAG consistently outperformed both the baseline and fine-tuned models in answer relevance and BLEU scores.
- Agentic RAG introduced slight performance drops but improved generalizability by handling irrelevant queries effectively.

---


## Future Work

1. Explore advanced RAG techniques like dense retrieval and cross-encoders.
2. Integrate real-time financial data sources (e.g., APIs, news feeds).
3. Develop a user-friendly interface for seamless interaction.
4. Implement robust safety measures for responsible AI usage in finance.
