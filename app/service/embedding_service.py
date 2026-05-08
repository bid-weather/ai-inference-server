import numpy as np
import torch
from app.model.kobert import KoBERTModel

def extract_embedding(text: str) -> np.ndarray:
    """
    공고 제목 텍스트를 받아서 KoBERT로 768차원 임베딩 벡터를 추출해 반환

    Args:
        text: 공고 제목 (예: "백엔드 개발자 채용")

    Returns:
        np.ndarray: 768차원 임베딩 벡터
    """
    
    tokenizer, model = KoBERTModel.get()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return embedding