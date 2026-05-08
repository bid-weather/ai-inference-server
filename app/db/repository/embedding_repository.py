import logging
import numpy as np
from sqlalchemy import text
from app.db.session import get_db

logger = logging.getLogger(__name__)

def save_embedding(announcement_id: int, embedding: np.array) -> None:
    """
    공고 제목 임베딩 벡터를 announcement_embedding 테이블에 저장
    이미 존재하면 업데이트 (upsert)

    Args:
        announcement_id: 공고 ID
        embedding: 768차원 numpy 배열
    """
    
    with get_db() as db:
        db.execute(
            text("""
                 INSERT INTO announcement_embedding (id, embedding)
                 VALUES (:id, :embedding)
                 ON CONFLICT (id)
                 DO UPDATE SET embedding = EXCLUDED.embedding
            """),
            {
                "id": announcement_id,
                "embedding": embedding.tolist()
            }
        )
        
        db.commit()

def find_most_similar_subcategory(embedding: np.ndarray) -> dict | None:
    """
    소분류를 위해 기분류 공고들과 코사인 유사도를 비교하여 가장 유사한 공고 반환

    Args:
        embedding: 비교할 768차원 numpy 배열

    Returns:
        dict | None: 가장 유사한 공고 정보 또는 None (기분류 공고 없을 때)
        {
            "subcategory_id": int,
            "similarity": float
        }
    """
    
    with get_db() as db:
        result = db.execute(
            text("""
                SELECT a.subcategory_id, 
                        1 - (ae.embedding <=> CAST(:embedding AS vector)) AS similarity
                FROM announcement_embedding ae
                JOIN announcement a ON a.id = ae.id
                WHERE a.subcategory_id IS NOT NULL
                ORDER BY ae.embedding <=> CAST(:embedding AS vector)
                LIMIT 1
            """),
            {"embedding": str(embedding.tolist())}
        ).fetchone()
    
    if result is None:
        return None
    
    return {
        "subcategory_id": result.subcategory_id,
        "similarity": float(result.similarity)
    }