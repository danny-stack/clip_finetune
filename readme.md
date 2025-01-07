# CLIP Fine-tuning Project

This repository contains code for fine-tuning OpenAI's CLIP (Contrastive Language-Image Pre-training) model on custom fashion datasets. CLIP, originally developed by OpenAI, learns visual concepts from natural language supervision and can be applied to various visual and language tasks.

## Overview

The project fine-tunes the `clip-vit-base-patch32` model on an Indo-fashion dataset to improve its performance on fashion-specific image-text matching tasks. We implement various training optimizations including:

- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Vision and text encoder backbone freezing

## Project Structure

```
CLIP_Finetune/
├── checkpoints/          # Saved model checkpoints
├── indo-fashion-dataset/ # Dataset files
├── runs/                 # Training logs
├── model_train.ipynb     # Training notebook
├── evaluate.py          # Evaluation script
├── earlystopping.py    # Early stopping implementation
└── retrieval_demo.py    # Demo for image-text retrieval
```

## Performance Metrics

| Metric | Value |
|--------|--------|
| I2T R@1 | 0.1659 |
| T2I R@1 | 0.1599 |
| I2T R@5 | 0.3944 |
| T2I R@5 | 0.3887 |
| I2T R@10 | 0.5117 |
| T2I R@10 | 0.5053 |
| Number of Epochs | 15 |
| Initial Learning Rate | 5e-5 |

## Key Features

- **MixFreeze Strategy**: Selective freezing of vision and text encoder backbones for optimal transfer learning
- **Efficient Training**: Implementation of gradient accumulation and mixed precision training
- **Evaluation Metrics**: Comprehensive evaluation using R@K metrics for both image-to-text and text-to-image retrieval
- **Early Stopping**: Prevents overfitting by monitoring validation metrics

## Install dependencies

```bash
pip install torch tqdm transformers clip PIL
```

## Evaluation

The evaluation script (`evaluate.py`) computes the following metrics:
- Image-to-Text Retrieval (I2T)
- Text-to-Image Retrieval (T2I)
- Recall@K (K=1,5,10)

## Usage

For evaluation:
```python
metrics = evaluate_model(model, test_loader, device)
print(metrics)
```

## References

1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. [A Guide to Fine-tuning CLIP Models with Custom Data](https://medium.com/aimonks/a-guide-to-fine-tuning-clip-models-with-custom-data-6c7c0d1416fb)

## License

This project is released under the MIT License.
