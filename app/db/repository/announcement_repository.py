import logging
import numpy as np
from sqlalchemy import text
from datetime import date
from app.db.session import get_db

logger = logging.getLogger(__name__)

def get_unclassified_announcements(start_date: date, end_date: date) -> list[dict]:
    """
    미분류 공고 전체 조회
    category_id 또는 subcategory_id가 NULL인 공고 조회

    Returns:
        list[dict]: 미분류 공고 목록
        [{"id": 1, "title": "공고 제목"}, ...]
    """

    with get_db() as db:
        results = db.execute(
            text("""
                SELECT id, title, category_id, subcategory_id
                FROM announcement
                WHERE (category_id IS NULL OR subcategory_id IS NULL)
                AND title IS NOT NULL
                AND title != ''
                AND date >= :start_date
                AND date < :end_date
                ORDER BY id
            """),
            {
                "start_date": start_date,
                "end_date": end_date,
            }
        ).fetchall()

    return [
        {
            "id": result.id, 
            "title": result.title, 
            "category_id": result.category_id, 
            "subcategory_id": result.subcategory_id,
        } for result in results
    ]

def update_category(
    announcement_id: int,
    category_id: int,
    subcategory_id: int
) -> None:
    """
    분류 결과를 announcement 테이블에 업데이트

    Args:
        announcement_id: 공고 ID
        category_id: 분류된 카테고리 ID
        subcategory_id: 분류된 서브카테고리 ID
    """

    with get_db() as db:
        db.execute(
            text("""
                UPDATE announcement
                SET
                    category_id = :category_id,
                    subcategory_id = :subcategory_id
                WHERE id = :id
            """),
            {
                "id": announcement_id,
                "category_id": category_id,
                "subcategory_id": subcategory_id
            }
        )
        db.commit()