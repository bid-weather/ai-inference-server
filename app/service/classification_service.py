import logging
import numpy as np
from datetime import date, timedelta
from app.db.repository.embedding_repository import save_embedding, find_most_similar_subcategory
from app.db.repository.announcement_repository import get_unclassified_announcements, update_category
from app.service.embedding_service import extract_embedding

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.85

CATEGORY_ANCHORS = {
    2: [
        "소프트웨어 개발 용역",
        "시스템 구축 및 운영",
        "IT 기술 지원 서비스",
        "네트워크 인프라 구축",
        "데이터베이스 구축 용역",
        "웹사이트 개발 및 유지보수",
        "모바일 앱 개발 용역",
        "클라우드 시스템 구축",
        "정보보안 시스템 구축",
        "AI 및 머신러닝 솔루션 개발",
        "ERP 시스템 구축 및 운영",
        "빅데이터 플랫폼 구축",
        "GIS 시스템 개발 용역",
        "영상관제 시스템 구축",
        "통신망 구축 및 유지보수",
        "SW 유지보수 및 기술지원",
        "사이버보안 취약점 점검",
        "디지털 전환 컨설팅",
        "스마트시티 플랫폼 구축",
        "전산장비 유지보수 용역",
    ],
    3: [
        "청소 및 환경미화 용역",
        "시설물 유지관리 용역",
        "경비 및 보안 용역",
        "행정 지원 서비스",
        "물품 배송 및 운반 용역",
        "건물 위생관리 용역",
        "주차관리 용역",
        "조경 유지관리 용역",
        "급식 및 식당 운영 용역",
        "세탁 및 린넨 관리 용역",
        "인쇄 및 문서 관리 용역",
        "통번역 서비스 용역",
        "콜센터 운영 용역",
        "행사 진행 및 의전 용역",
        "폐기물 수거 및 처리 용역",
        "소독 및 방역 용역",
        "건축물 안전점검 용역",
        "회계 및 세무 지원 용역",
        "홍보 및 마케팅 용역",
        "교육 훈련 위탁 용역",
    ],
}

_anchor_embeddings: dict[int, list[np.ndarray]] | None = None

def _get_anchor_embeddings() -> dict[int, list[np.ndarray]]:
    global _anchor_embeddings
    if _anchor_embeddings is not None:
        return _anchor_embeddings
    
    logger.info("카테고리 anchor 임베딩 초기화 중...")
    _anchor_embeddings = {
        category_id: [extract_embedding(s) for s in sentences]
        for category_id, sentences in CATEGORY_ANCHORS.items()
    }
    logger.info("카테고리 anchor 임베딩 초기화 완료")
    return _anchor_embeddings

def _classify_category(embedding: np.ndarray) -> int:
    anchor_embeddings = _get_anchor_embeddings()
    
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    best_category_id = None
    best_avg_similarity = -1.0
    
    for category_id, anchors in anchor_embeddings.items():
        avg_similarity = np.mean([cosine_similarity(embedding, anchor) for anchor in anchors])
        if avg_similarity > best_avg_similarity:
            best_avg_similarity = avg_similarity
            best_category_id = category_id
    
    return category_id

def classify_announcement(
    announcement_id: int,
    title: str,
    embedding: np.ndarray,
    category_id: int | None,
    subcategory_id: int | None
) -> dict:
    """
    공고 임베딩을 저장하고 기분류 공고와 유사도 비교하여 분류

    Args:
        announcement_id: 공고 ID
        title: 공고 제목 (로깅용)
        embedding: extract_embedding()으로 추출한 768차원 벡터
        category_id: 대분류 id | None
        subcategory_id: 소분류 id | None

    Returns:
        dict: 분류 결과
        {
            "announcement_id": int,
            "category_id": int,
            "subcategory_id": int,
            "similarity": float,
            "classified": boolean
        }
    """
    
    save_embedding(announcement_id, embedding)
    
    if category_id is None:
        category_id = _classify_category(embedding)
        
    if subcategory_id is not None:
        update_category(
            announcement_id=announcement_id,
            category_id=category_id,
            subcategory_id=subcategory_id
        )
        return {
            "announcement_id": announcement_id,
            "category_id": category_id,
            "subcategory_id": subcategory_id,
            "similarity": None,
            "classified": True
        }
        
    result = find_most_similar_subcategory(embedding)
    
    if result is None:
        logger.warning(f"유사 공고 없음 또는 유사도 미달: id={announcement_id}, title={title[:20]}, ")
        logger.warning(f"similarity={result['similarity'] if result else None}")
        return {
            "announcement_id": announcement_id,
            "category_id": category_id,
            "subcategory_id": None,
            "similarity": result["similarity"] if result else None,
            "classified": False
        }
    
    subcategory_id = result["subcategory_id"]
    similarity = result["similarity"]
    
    update_category(
        announcement_id=announcement_id,
        category_id=category_id,
        subcategory_id=subcategory_id
    )
    
    return {
        "announcement_id": announcement_id,
        "category_id": category_id,
        "subcategory_id": subcategory_id,
        "similarity": similarity,
        "classified": True
    }

def classify_all_unclassified(start_date: date):
    """
    날짜 단위 chunk로 미분류 공고 전체 분류 (배치 스크립트 / Kafka 시그널 양쪽에서 공용)
    """
    total = 0
    success = 0
    fail = 0
    
    current = start_date
    yesterday = date.today() - timedelta(days=1)
    
    while current <= yesterday:
        next_day = current + timedelta(days=1)
        
        announcements = get_unclassified_announcements(
            start_date=current,
            end_date=next_day,
        )
        
        if not announcements:
            logger.info(f"[{current}] 미분류 공고 없음, 스킵")
            current = next_day
            continue
        
        logger.info(f"[{current}] {len(announcements)}건 처리 시작")
        
        for announcement in announcements:
            try:
                embedding = extract_embedding(announcement["title"])
                result = classify_announcement(
                    announcement_id=announcement["id"],
                    title=announcement["title"],
                    embedding=embedding,
                    category_id=announcement.get("category_id"),
                    subcategory_id=announcement.get("subcategory_id")
                    
                )
                if result["classified"]:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                logger.error(f"분류 실패: announcement_id={announcement["id"]}, error={e}", exc_info=True)
            
        logger.info(f"[{current}] 완료")
        current = next_day
    
    summary = {"total": total, "success": success, "fail": fail}
    logger.info(f"전체 완료: {summary}")