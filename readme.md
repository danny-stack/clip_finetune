# CLIP Fine-tuning Project

This repository contains code for fine-tuning OpenAI's CLIP (Contrastive Language-Image Pre-training) model on custom fashion datasets. CLIP, originally developed by OpenAI, learns visual concepts from natural language supervision and can be applied to various visual and language tasks.

## Overview

The project fine-tunes the `clip-vit-base-patch32` model on an Indo-fashion dataset to improve its performance on fashion-specific image-text matching tasks. We implement various training optimizations including:

- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Vision and text encoder backbone freezing


## Performance Metrics

| Model | I2T R@1 | T2I R@1 | I2T R@5 | T2I R@5 | I2T R@10 | T2I R@10 | Epochs | Learning Rate | Description |
|-------|---------|---------|---------|---------|-----------|-----------|---------|---------------|-------------|
| Zero-shot | 0.0351 | 0.0220 | 0.1085 | 0.0707 | 0.1557 | 0.1117 | / | / | Original CLIP without fine-tuning |
| Vision Freeze | 0.1924 | 0.1873 | 0.4407 | 0.4397 | 0.5607 | 0.5613 | 15 | 5e-5 | Freeze first 6 layers of vision encoder |
| MixFreeze | 0.1659 | 0.1599 | 0.3944 | 0.3887 | 0.5117 | 0.5053 | 15 | 5e-5 | Freeze backbone of both encoders |
| Unfreeze | 0.1652 | 0.1548 | 0.4007 | 0.3861 | 0.5171 | 0.5160 | 20 | 5e-5 | Finetune all the layers |


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


## References

1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. [A Guide to Fine-tuning CLIP Models with Custom Data](https://medium.com/aimonks/a-guide-to-fine-tuning-clip-models-with-custom-data-6c7c0d1416fb)

## License

This project is released under the MIT License.
