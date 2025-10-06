from transformers import CLIPModel, CLIPProcessor
import torch

# Load CLIP model and tokenizer
device = "cuda"

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

def compute_image_embedding(image):
    inputs = clip_processor(
        text=[""],  # Placeholder for image-only processing
        images=image,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(inputs["pixel_values"])
    return image_embeds / image_embeds.norm(dim=1, keepdim=True)


def compute_text_embedding(text):
    inputs = clip_processor(
        text=[text] if isinstance(text, str) else text,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    return text_embeds / text_embeds.norm(dim=1, keepdim=True)


def clip_score(c_embedding: torch.Tensor, v_embedding: torch.Tensor, w: float = 2.5) -> float:
    """
    Compute the CLIP-Score for a given candidate caption and image using PyTorch.
    
    Args:
        c_embedding (torch.Tensor): CLIP embedding of the candidate caption (1D tensor).
        v_embedding (torch.Tensor): CLIP embedding of the image (1D tensor).
        w (float): Weight scaling factor (default is 2.5).
        
    Returns:
        float: The CLIP-Score.
    """
    # Compute cosine similarity
    cos_similarity = torch.nn.functional.cosine_similarity(c_embedding, v_embedding, dim=1)
    # Apply rescaling and max(0, cos_similarity)
    return w * max(cos_similarity.item(), 0)

def refclip_score(
    c_embedding: torch.Tensor,
    r_embeddings: list[torch.Tensor],
    v_embedding: torch.Tensor,
    w: float = 2.5
) -> float:
    """
    Compute the RefCLIPScore for a given candidate caption, image, and reference captions using PyTorch.
    
    Args:
        c_embedding (torch.Tensor): CLIP embedding of the candidate caption (1D tensor).
        r_embeddings (List[torch.Tensor]): List of CLIP embeddings of reference captions (1D tensors).
        v_embedding (torch.Tensor): CLIP embedding of the image (1D tensor).
        w (float): Weight scaling factor (default is 2.5).
        
    Returns:
        float: The RefCLIPScore.
    """
    # Compute CLIP-S(c, v)
    clip_s = clip_score(c_embedding, v_embedding, w=w)
    
    # Compute max cosine similarity between candidate and references
    max_ref_similarity = max(
        torch.nn.functional.cosine_similarity(c_embedding, r, dim=1).item()
        for r in r_embeddings
    )
    
    # Compute harmonic mean
    if clip_s > 0 and max_ref_similarity > 0:
        return 2 * clip_s * max_ref_similarity / (clip_s + max_ref_similarity)
    else:
        return 0.0
    
def compute_metrics(output, target, image_patch):
    # Compute embeddings
    output_embedding = compute_text_embedding(output)
    target_embedding = compute_text_embedding(target)
    image_embedding = compute_image_embedding(image_patch)
    
    # Compute scores
    ref_clip_score = refclip_score(output_embedding, [target_embedding], image_embedding)
    text_similarity_score = torch.nn.functional.cosine_similarity(output_embedding, target_embedding, dim=1).item()
    
    return ref_clip_score, text_similarity_score
    