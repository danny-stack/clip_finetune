import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    """
    Compute image-to-text and text-to-image retrieval accuracy（R@1, R@5, R@10）
    """
    model.eval()
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Extracting features"):
            images = images.to(device)
            texts = texts.to(device)
            
            outputs = model(input_ids=texts, pixel_values=images)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            all_image_features.append(image_features)
            all_text_features.append(text_features)
    
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)

    similarity = (image_features @ text_features.T)
    
    metrics = {}
    for k in [1, 5, 10]:
        i2t_recall = compute_recall_at_k(similarity, k)
        t2i_recall = compute_recall_at_k(similarity.T, k)
        metrics[f'I2T_R@{k}'] = i2t_recall
        metrics[f'T2I_R@{k}'] = t2i_recall
    
    return metrics

def compute_recall_at_k(similarity, k):
    """Compute Recall@K"""
    batch_size = similarity.shape[0]
    indices = torch.topk(similarity, k, dim=1)[1]
    targets = torch.arange(batch_size).view(-1, 1).to(similarity.device)
    correct = (indices == targets)
    return (correct.sum(1) > 0).float().mean().item()