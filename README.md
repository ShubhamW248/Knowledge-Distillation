# Knowledge Distillation for Text Classification

This project explores the use of **knowledge distillation** to train compact and efficient language models for sentiment classification. Specifically, we distill the knowledge of a large BERT model (teacher) into a smaller DistilBERT model (student), aiming to retain accuracy while improving computational efficiency.

## Overview

Transformer-based models like BERT have demonstrated state-of-the-art performance on a wide range of NLP tasks. However, their large size and high latency make them impractical for real-time or on-device applications. Knowledge distillation addresses this issue by training a smaller model to mimic the behavior of a larger one.

In this project:

- We fine-tuned a **BERT-base-uncased** model on the Financial Phrasebank dataset for sentiment classification.
- We then trained a **DistilBERT-base-uncased** student model using both the ground truth labels and the teacher model’s soft predictions.
- Finally, we evaluated both models on accuracy, model size, and inference speed.

## Dataset

The dataset used is the [Financial Phrasebank](https://huggingface.co/datasets/financial_phrasebank), a collection of financial news statements labeled as **positive**, **negative**, or **neutral**. It’s widely used for financial sentiment analysis and presents a meaningful challenge for text classification models.

## Methodology

The teacher model was first fine-tuned using supervised learning. The student model was then trained using a combined loss function:

- **Hard Target Loss**: Cross-entropy between the student’s predictions and true labels.
- **Soft Target Loss**: Kullback-Leibler divergence between the student’s and teacher’s predicted distributions.

This dual-target approach allows the student to benefit both from explicit supervision and from the generalization learned by the teacher.

## Results

| Metric               | Teacher (BERT)     | Student (DistilBERT) |
|----------------------|--------------------|-----------------------|
| Accuracy             | 94.41%             | 95.29%                |
| Model Size           | 417.65 MB          | 255.42 MB             |
| Inference Time       | 7.66 ms/sample     | 3.98 ms/sample        |

### Interpretation

The student model achieved slightly higher accuracy than the teacher on the test set. While this might seem surprising, such results can occasionally occur due to stochastic factors in training or mild overfitting on the teacher’s side. In general, the objective of distillation is to closely match—not exceed—the teacher’s performance.

What’s more significant is that the student model delivers this performance with **nearly 40% less memory usage** and **almost 2x faster inference**, making it far more efficient for real-world applications.

## Conclusion

This project demonstrates the effectiveness of knowledge distillation for compressing large transformer models in NLP. The student model preserved the performance of the teacher while significantly reducing the computational footprint, validating distillation as a practical strategy for deploying language models in production environments.

The approach shown here can be applied to other tasks and model architectures to achieve similar trade-offs between performance and efficiency.
