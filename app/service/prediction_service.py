import logging
from datetime import date, timedelta

from app.db.repository.prediction_repository import (
    get_announcement_counts_by_category,
    get_announcement_counts_by_cat_sub,
    get_weather_actual,
    get_weather_avg_past_two_years,
    get_calendar,
    insert_predictions,
)
from app.model.prediction_model import (
    PredictionModel, predict_one, pred_end_date, prev_month_next_day, _coerce_weather,
)

logger = logging.getLogger(__name__)


def predict_and_save(predicted_at: date) -> None:
    """
    추론 파이프라인 진입점.

    Args:
        predicted_at: 예측 기준일 (Kafka 메시지로 수신한 날짜)

    저장 범위: predicted_at ~ 익월 전일
    lag 소스 : announcement 실제값 → auto-regressive cache
    """
    cat_booster     = PredictionModel.cat_booster
    cat_sub_booster = PredictionModel.cat_sub_booster
    cat_unit_ids    = PredictionModel.cat_unit_ids
    cat_sub_unit_ids = PredictionModel.cat_sub_unit_ids
    train_days      = PredictionModel.train_days

    logger.info(f"[PredictionService] Start inference. predicted_at={predicted_at}")

    # ── 1. 날짜 범위 계산 ──────────────────────────────────────────
    lag_source_start = prev_month_next_day(predicted_at)
    pred_end         = pred_end_date(predicted_at)

    pred_dates   = [predicted_at + timedelta(days=i) for i in range((pred_end - predicted_at).days + 1)]
    future_dates = [d for d in pred_dates if d > predicted_at]

    logger.info(
        f"[PredictionService] lag_source={lag_source_start}~{predicted_at - timedelta(days=1)}, "
        f"predict={predicted_at}~{pred_end}, train_days={train_days}"
    )

    # ── 2. 공고 실제값 수집 (lag 소스) ────────────────────────────
    cat_counts     = get_announcement_counts_by_category(lag_source_start, predicted_at)
    cat_sub_counts = get_announcement_counts_by_cat_sub(lag_source_start, predicted_at)

    # ── 3. 캘린더 수집 ─────────────────────────────────────────────
    calendar_map = get_calendar(predicted_at, pred_end)

    # ── 4. 날씨 맵 구성 ────────────────────────────────────────────
    actual_weather_list = get_weather_actual(lag_source_start, predicted_at)
    actual_weather: dict = {r["date"]: r for r in actual_weather_list}
    # predicted_at 당일 실제 날씨가 DB에 없을 수 있으므로 과거 2년 평균을 fallback으로 사용.
    # actual_weather에 있으면 실제값이 덮어씀.
    fallback_dates = [predicted_at, *future_dates]
    future_weather = get_weather_avg_past_two_years(fallback_dates)  # None 포함 가능
    weather_map: dict = {**future_weather, **actual_weather}

    # ── 5. lag 소스 구성 ──────────────────────────────────────────
    # category: (date, category_id) → count
    cat_actual: dict[tuple, float] = {
        (r["date"], r["category_id"]): float(r["count"]) for r in cat_counts
    }
    # cat×sub: (date, composite_unit_id) → count
    cat_sub_actual: dict[tuple, float] = {
        (r["date"], r["category_id"] * 10_000 + r["subcategory_id"]): float(r["count"])
        for r in cat_sub_counts
    }

    # ── 6. Auto-regressive 추론 ────────────────────────────────────
    cat_cache:     dict[tuple, float] = {}
    cat_sub_cache: dict[tuple, float] = {}
    result_rows:   list[dict]         = []
    skipped_dates: list[date]         = []

    for d in pred_dates:
        weather  = weather_map.get(d)
        calendar = calendar_map.get(d)

        if calendar is None:
            logger.warning(f"[PredictionService] 캘린더 결측 — date={d}. 추론 skip.")
            skipped_dates.append(d)
            continue

        weather = _coerce_weather(weather)

        cat_lag_map     = {**cat_actual,     **cat_cache}
        cat_sub_lag_map = {**cat_sub_actual, **cat_sub_cache}

        # category 예측 (unit_id = category_id)
        for unit_id in cat_unit_ids:
            count = predict_one(cat_booster, d, unit_id, weather, calendar, cat_lag_map)
            cat_cache[(d, unit_id)] = float(count)
            result_rows.append({
                "date":           d,
                "category_id":    unit_id,
                "subcategory_id": None,
                "count":          count,
                "predicted_at":   predicted_at,
            })

        # cat×sub 예측 (unit_id = category_id * 10_000 + subcategory_id)
        for unit_id in cat_sub_unit_ids:
            cat_id = unit_id // 10_000
            sub_id = unit_id % 10_000
            count = predict_one(cat_sub_booster, d, unit_id, weather, calendar, cat_sub_lag_map)
            cat_sub_cache[(d, unit_id)] = float(count)
            result_rows.append({
                "date":           d,
                "category_id":    cat_id,
                "subcategory_id": sub_id,
                "count":          count,
                "predicted_at":   predicted_at,
            })

    if skipped_dates:
        logger.warning(f"[PredictionService] Skip된 날짜 {len(skipped_dates)}일: {skipped_dates}")

    # ── 7. DB 저장 ─────────────────────────────────────────────────
    insert_predictions(result_rows)
    logger.info(
        f"[PredictionService] Done. total={len(result_rows)} rows saved, "
        f"skipped={len(skipped_dates)} dates."
    )