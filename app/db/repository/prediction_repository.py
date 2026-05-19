import logging
from datetime import date
from sqlalchemy import text
from app.db.session import get_db

logger = logging.getLogger(__name__)


def get_announcement_counts_by_category(start_date: date, end_date: date) -> list[dict]:
    """
    날짜별 category_id 공고 수 집계 (subcategory_id NULL 포함 전체)

    Returns:
        [{"date": date, "category_id": int, "count": int}, ...]
    """
    with get_db() as db:
        results = db.execute(
            text("""
                SELECT date, category_id, COUNT(*) AS count
                FROM announcement
                WHERE category_id IS NOT NULL
                  AND date >= :start_date
                  AND date < :end_date
                GROUP BY date, category_id
                ORDER BY date, category_id
            """),
            {"start_date": start_date, "end_date": end_date}
        ).fetchall()

    return [
        {"date": row.date, "category_id": row.category_id, "count": row.count}
        for row in results
    ]


def get_announcement_counts_by_cat_sub(start_date: date, end_date: date) -> list[dict]:
    """
    날짜별 category_id × subcategory_id 조합 공고 수 집계

    Returns:
        [{"date": date, "category_id": int, "subcategory_id": int, "count": int}, ...]
    """
    with get_db() as db:
        results = db.execute(
            text("""
                SELECT date, category_id, subcategory_id, COUNT(*) AS count
                FROM announcement
                WHERE category_id IS NOT NULL
                  AND subcategory_id IS NOT NULL
                  AND date >= :start_date
                  AND date < :end_date
                GROUP BY date, category_id, subcategory_id
                ORDER BY date, category_id, subcategory_id
            """),
            {"start_date": start_date, "end_date": end_date}
        ).fetchall()

    return [
        {
            "date": row.date,
            "category_id": row.category_id,
            "subcategory_id": row.subcategory_id,
            "count": row.count,
        }
        for row in results
    ]


def get_cat_sub_pairs() -> list[tuple[int, int]]:
    """
    실제 데이터에 존재하는 (category_id, subcategory_id) 조합 반환.
    매핑 테이블 대신 announcement 테이블에서 직접 집계.

    Returns:
        [(category_id, subcategory_id), ...]
    """
    with get_db() as db:
        results = db.execute(
            text("""
                SELECT DISTINCT category_id, subcategory_id
                FROM announcement
                WHERE category_id IS NOT NULL
                  AND subcategory_id IS NOT NULL
                ORDER BY category_id, subcategory_id
            """)
        ).fetchall()

    return [(row.category_id, row.subcategory_id) for row in results]


def get_calendar(start_date: date, end_date: date) -> dict[date, dict]:
    """
    calendar_date 테이블에서 날짜별 is_weekend, is_holiday 조회

    Returns:
        { date: {"is_weekend": bool, "is_holiday": bool} }
    """
    with get_db() as db:
        results = db.execute(
            text("""
                SELECT date, is_weekend, COALESCE(is_holiday, FALSE) AS is_holiday
                FROM calendar_date
                WHERE date >= :start_date
                  AND date <= :end_date
                ORDER BY date
            """),
            {"start_date": start_date, "end_date": end_date}
        ).fetchall()

    return {
        row.date: {
            "is_weekend": bool(row.is_weekend),
            "is_holiday": bool(row.is_holiday),
        }
        for row in results
    }


def get_weather_actual(start_date: date, end_date: date) -> list[dict]:
    """
    실제 날씨 데이터 조회 (start_date ~ end_date 포함).
    NULL 컬럼은 None으로 반환 — 호출부에서 결측 여부를 판단.

    Returns:
        [{"date": date, "max_temp": float | None, ...}, ...]
    """
    with get_db() as db:
        results = db.execute(
            text("""
                SELECT
                    date,
                    max_temp,
                    min_temp,
                    precipitation,
                    snow,
                    humidity,
                    daily_max_wind_speed
                FROM weather_daily
                WHERE date >= :start_date
                  AND date <= :end_date
                ORDER BY date
            """),
            {"start_date": start_date, "end_date": end_date}
        ).fetchall()

    def _to_float(v) -> float | None:
        return float(v) if v is not None else None

    return [
        {
            "date": row.date,
            "max_temp": _to_float(row.max_temp),
            "min_temp": _to_float(row.min_temp),
            "precipitation": _to_float(row.precipitation),
            "snow": _to_float(row.snow),
            "humidity": _to_float(row.humidity),
            "daily_max_wind_speed": _to_float(row.daily_max_wind_speed),
        }
        for row in results
    ]


def get_weather_avg_past_two_years(target_dates: list[date]) -> dict[date, dict | None]:
    """
    미래 날짜에 대해 전년·전전년 동일 날짜의 날씨 평균 계산.
    두 해 중 하나라도 데이터가 없으면 None 반환 — 호출부에서 skip 처리.

    Returns:
        { target_date: {"max_temp": float, ...} | None }
    """
    if not target_dates:
        return {}

    past_dates = []
    for d in target_dates:
        for years_back in (1, 2):
            past_dates.append(d.replace(year=d.year - years_back))

    with get_db() as db:
        results = db.execute(
            text("""
                SELECT
                    date,
                    max_temp,
                    min_temp,
                    precipitation,
                    snow,
                    humidity,
                    daily_max_wind_speed
                FROM weather_daily
                WHERE date = ANY(:dates)
                ORDER BY date
            """),
            {"dates": past_dates}
        ).fetchall()

    def _to_float(v) -> float | None:
        return float(v) if v is not None else None

    past_weather: dict[date, dict] = {}
    for row in results:
        past_weather[row.date] = {
            "max_temp": _to_float(row.max_temp),
            "min_temp": _to_float(row.min_temp),
            "precipitation": _to_float(row.precipitation),
            "snow": _to_float(row.snow),
            "humidity": _to_float(row.humidity),
            "daily_max_wind_speed": _to_float(row.daily_max_wind_speed),
        }

    weather_cols = ["max_temp", "min_temp", "precipitation", "snow", "humidity", "daily_max_wind_speed"]
    result_map: dict[date, dict | None] = {}

    for d in target_dates:
        year_records = [
            past_weather.get(d.replace(year=d.year - years_back))
            for years_back in (1, 2)
        ]
        # 둘 중 하나라도 없으면 None
        if any(r is None for r in year_records):
            missing = [
                str(d.replace(year=d.year - yb))
                for yb, r in zip((1, 2), year_records)
                if r is None
            ]
            logger.warning(
                f"[PredictionRepository] 날씨 데이터 없음 — target={d}, "
                f"missing_past_dates={missing}. 해당 날짜 추론 skip 예정."
            )
            result_map[d] = None
            continue

        # 컬럼별 평균 (각 record의 None 컬럼도 처리)
        averaged: dict[str, float | None] = {}
        for col in weather_cols:
            vals = [r[col] for r in year_records]  # type: ignore[index]
            if any(v is None for v in vals):
                averaged[col] = None
            else:
                averaged[col] = sum(vals) / len(vals)  # type: ignore[arg-type]

        result_map[d] = averaged

    return result_map


def insert_predictions(rows: list[dict]) -> None:
    """
    prediction_daily 테이블에 예측 결과 bulk INSERT

    Args:
        rows: [
            {
                "date": date,
                "category_id": int,
                "subcategory_id": int | None,
                "count": int,
                "predicted_at": date,
            },
            ...
        ]
    """
    if not rows:
        return

    predicted_at = rows[0]["predicted_at"]

    with get_db() as db:
        db.execute(
            text("DELETE FROM prediction_daily WHERE predicted_at = :predicted_at"),
            {"predicted_at": predicted_at}
        )
        db.execute(
            text("""
                INSERT INTO prediction_daily
                    (date, category_id, subcategory_id, count, predicted_at)
                VALUES
                    (:date, :category_id, :subcategory_id, :count, :predicted_at)
            """),
            rows
        )
        db.commit()

    logger.info(f"[PredictionRepository] Inserted {len(rows)} rows into prediction_daily.")