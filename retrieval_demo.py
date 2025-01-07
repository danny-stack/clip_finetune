def V2T_retrieval(model, processor, image_path, text_queries, k=5):
    """Retrieval text from image"""
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(inputs['pixel_values'].to(device))

    all_text_features = []
    for text_query in text_queries:
        inputs = processor.tokenizer(text=text_query, return_tensors="pt", truncation=True)
        # text_features = model.get_text_features(inputs['pixel_values'].to(device))
        text_features = model.get_text_features(**inputs.to(device))
        all_text_features.append(text_features)

    all_text_features = torch.cat(all_text_features)
    similarities = image_features @ all_text_features.T
    top_indices = similarities.topk(k).indices

    return [text_queries[idx] for idx in top_indices[0]]

def T2V_retrieval(model, processor, text_query, image_folder,k=10):
    """Retrieval image from text"""
    model.eval()
    model.to(device)
    
    inputs = processor.tokenizer(text=text_query, return_tensors="pt", truncation=True)
    text_features = model.get_text_features(**inputs.to(device))

    all_image_features = []
    all_image_paths = []
    for img_path in tqdm(glob.glob(f"{image_folder}/*.jpg")):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(inputs['pixel_values'].to(device))
            all_image_features.append(image_features)
            all_image_paths.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    all_image_features = torch.cat(all_image_features)
    similarities = text_features @ all_image_features.T
    top_indices = similarities.topk(k).indices

    return [all_image_paths[idx] for idx in top_indices[0]]