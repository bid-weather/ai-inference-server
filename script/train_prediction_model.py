"""
학습 실행 스크립트

사용법:
    python -m script.train_prediction_model \
        --start-date 2026-01-01 \
        --end-date   2026-05-19

각 기준일을 origin으로, 그 날부터 익월 전일까지 auto-regressive 방식으로
(feature, label) 쌍을 수집한 뒤 LightGBM 모델 2개를 학습.

실행 결과:
    config.model_weights_dir 경로에 아래 파일 저장
    ├── category_model.txt
    ├── cat_sub_model.txt
    └── meta.json
"""

import argparse
import logging
import sys
from datetime import date, timedelta

sys.path.append(".")

from app.core.config import settings
from app.db.repository.prediction_repository import (
    get_announcement_counts_by_category,
    get_announcement_counts_by_cat_sub,
    get_weather_actual,
    get_calendar,
)
from app.model.prediction_model import (
    LAG_DAYS,
    collect_training_rows,
    pred_end_date,
    prev_month_next_day,
    train,
    save_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(start_date: date, end_date: date) -> None:
    max_lag    = max(LAG_DAYS)
    target_end = pred_end_date(end_date)
    lag_start  = prev_month_next_day(start_date)

    logger.info(
        f"[Train] origin={start_date}~{end_date}, "
        f"target_end={target_end}, lag_start={lag_start}"
    )

    # ── 1. 공고 수 수집 ─────────────────────────────────────────────
    # lag_start ~ target_end 를 한 번에 조회 (lag 소스 + 학습 대상 통합)
    all_cat_counts     = get_announcement_counts_by_category(lag_start, target_end + timedelta(days=1))
    all_cat_sub_counts = get_announcement_counts_by_cat_sub(lag_start, target_end + timedelta(days=1))

    if not all_cat_counts:
        logger.error("[Train] category 공고 데이터 없음. 종료.")
        sys.exit(1)
    if not all_cat_sub_counts:
        logger.error("[Train] cat_sub 공고 데이터 없음. 종료.")
        sys.exit(1)

    logger.info(
        f"[Train] Loaded category rows={len(all_cat_counts)}, "
        f"cat_sub rows={len(all_cat_sub_counts)}"
    )

    # ── 2. 날씨·캘린더 수집 ────────────────────────────────────────
    weather_list = get_weather_actual(lag_start, target_end)
    weather_map  = {r["date"]: r for r in weather_list}
    calendar_map = get_calendar(start_date, target_end)

    # ── 3. actual_map 구성 ─────────────────────────────────────────
    cat_actual_map: dict[tuple, float] = {
        (r["date"], r["category_id"]): float(r["count"])
        for r in all_cat_counts
    }
    cat_sub_actual_map: dict[tuple, float] = {
        (r["date"], r["category_id"] * 10_000 + r["subcategory_id"]): float(r["count"])
        for r in all_cat_sub_counts
    }

    # ── 4. unit_id 결정 (학습 기간 start_date~target_end 에 등장한 전체) ──
    cat_unit_ids = sorted({
        r["category_id"]
        for r in all_cat_counts
        if start_date <= r["date"] <= target_end
    })
    cat_sub_unit_ids = sorted({
        r["category_id"] * 10_000 + r["subcategory_id"]
        for r in all_cat_sub_counts
        if start_date <= r["date"] <= target_end
    })

    logger.info(
        f"[Train] cat_unit_ids={len(cat_unit_ids)}, "
        f"cat_sub_unit_ids={len(cat_sub_unit_ids)}"
    )

    # ── 5. origin_dates 생성 ───────────────────────────────────────
    origin_dates = [
        start_date + timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]

    # ── 6. Auto-regressive 학습 데이터 수집 ──────────────────────
    logger.info(f"[Train] Collecting rows for {len(origin_dates)} origin dates...")

    cat_X, cat_y         = collect_training_rows(
        origin_dates, cat_unit_ids, cat_actual_map, weather_map, calendar_map
    )
    cat_sub_X, cat_sub_y = collect_training_rows(
        origin_dates, cat_sub_unit_ids, cat_sub_actual_map, weather_map, calendar_map
    )

    logger.info(
        f"[Train] Matrix — category: {cat_X.shape}, cat_sub: {cat_sub_X.shape}"
    )

    if cat_X.shape[0] == 0:
        logger.error("[Train] category 학습 row 없음. 종료.")
        sys.exit(1)
    if cat_sub_X.shape[0] == 0:
        logger.error("[Train] cat_sub 학습 row 없음. 종료.")
        sys.exit(1)

    # ── 7. 학습 ───────────────────────────────────────────────────
    logger.info("[Train] Training category model...")
    cat_model = train(cat_X, cat_y)

    logger.info("[Train] Training cat_sub model...")
    cat_sub_model = train(cat_sub_X, cat_sub_y)

    # ── 8. 저장 ───────────────────────────────────────────────────
    save_models(
        weights_dir=settings.model_weights_dir,
        cat_model=cat_model,
        cat_sub_model=cat_sub_model,
        cat_unit_ids=cat_unit_ids,
        cat_sub_unit_ids=cat_sub_unit_ids,
        train_days=max_lag,
    )

    logger.info(
        f"[Train] Done. "
        f"category units={len(cat_unit_ids)}, cat_sub units={len(cat_sub_unit_ids)}, "
        f"rows: cat={cat_X.shape[0]}, cat_sub={cat_sub_X.shape[0]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction model training script")
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        required=True,
        help="학습 시작 기준일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        required=True,
        help="학습 종료 기준일 (YYYY-MM-DD, 포함)",
    )
    args = parser.parse_args()
    main(start_date=args.start_date, end_date=args.end_date)
