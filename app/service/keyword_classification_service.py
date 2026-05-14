import logging
from datetime import date
from sqlalchemy import text
from app.db.session import get_db

logger = logging.getLogger(__name__)

KEYWORD_RULES: list[tuple[int, list[str]]] = [
    (1,  ['침수', '홍수', '배수', '우수관', '빗물', '방수', '제방', '하천정비', '유수지', '수방', '저류지', '펌프장']),
    (2,  ['긴급복구', '재해복구', '피해복구', '응급복구', '재난복구', '수해복구', '태풍복구', '복구공사', '재해대책']),
    (3,  ['폭염', '무더위', '쿨링', '그늘막', '열사병', '온열']),
    (4,  ['가뭄', '농업용수', '공업용수', '생활용수', '저수지', '취수', '긴급급수', '제한급수', '물부족', '관개', '수리시설', '양수장']),
    (5,  ['제설', '대설', '염화칼슘', '눈치우', '적설', '빙판', '결빙', '제빙', '모래살포']),
    (6,  ['동파', '혹한', '보온', '동절기', '한파', '방한', '보냉']),
    (7,  ['사면', '지반', '옹벽', '절개지', '토사', '산사태', '급경사', '석축', '토석류']),
    (8,  ['산불', '수목관리', '수목제거', '수목전지', '벌목', '방화선', '임도', '숲가꾸', '가로수', '산림']),
    (9,  ['기상재해', '재해영향평가', '재해위험지구', '풍수해저감', '재해저감', '자연재해대책', '풍수해보험']),
    (10, ['재난경보', '재난문자', '사이렌', '경보시스템', '재난통신']),
    (11, ['어업', '항만', '방파제', '어항', '해일', '어장']),
    (12, ['미세먼지', '황사', '대기오염', '대기질', '집진']),
    (13, ['방역', '감염병', '소독', '검역', '해충', '구제역', '조류독감']),
    (14, ['농작물', '축산', '방제', '병충해', '가축', '살충']),
    (15, ['수질', '정수', '하수', '오수', '폐수', '수처리', '정화조', '하수관', '오염수']),
    (16, ['포트홀', '싱크홀', '도로함몰', '지하공동', '도로파손', '노면', '아스팔트', '포장보수', '도로보수']),
]

def classify_one_day(d: date) -> None:
    with get_db() as db:
        try:
            for subcategory_id, keywords in KEYWORD_RULES:
                conditions = " OR ".join([
                    f"title LIKE :kw{i}"
                    for i, _ in enumerate(keywords)
                ])

                params = {
                    "d": d,
                    "sub_id": subcategory_id,
                    **{f"kw{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
                }

                db.execute(
                    text(f"""
                        UPDATE announcement
                        SET subcategory_id = :sub_id
                        WHERE date = :d
                          AND subcategory_id IS NULL
                          AND ({conditions})
                    """),
                    params,
                )

            db.commit()
            logger.info(f"키워드 분류 완료: {d}")

        except Exception as e:
            db.rollback()
            logger.error(f"키워드 분류 실패: {d}, 사유: {e}", exc_info=True)