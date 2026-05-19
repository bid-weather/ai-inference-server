import json
import logging
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from app.core.config import settings

logger = logging.getLogger(__name__)

LAG_DAYS = [7, 14]

WEATHER_COLS = [
    "max_temp",
    "min_temp",
    "precipitation",
    "snow",
    "humidity",
    "daily_max_wind_speed",
]

FEATURE_COLS = [
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "is_holiday",
    *WEATHER_COLS,
    "lag_7d",
    "lag_14d",
    "unit_id",  # category_id 또는 cat_sub 복합키 — label encoding 없이 정수 그대로 사용
]

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 300,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

CATEGORY_MODEL_FILE = "category_model.txt"
CAT_SUB_MODEL_FILE  = "cat_sub_model.txt"
META_FILE           = "meta.json"


def pred_end_date(d: date) -> date:
    """예측 종료일: d로부터 익월의 같은 날 - 1일."""
    if d.month == 12:
        next_month_first = date(d.year + 1, 1, 1)
    else:
        next_month_first = date(d.year, d.month + 1, 1)
    return next_month_first + timedelta(days=d.day - 2)


def prev_month_next_day(d: date) -> date:
    """전월 익일: 전월의 같은 날 + 1일."""
    import calendar as _cal
    prev_month = 12 if d.month == 1 else d.month - 1
    prev_year  = d.year - 1 if d.month == 1 else d.year
    max_day    = _cal.monthrange(prev_year, prev_month)[1]
    return date(prev_year, prev_month, min(d.day, max_day)) + timedelta(days=1)


def _coerce_weather(weather: dict | None) -> dict:
    """날씨 dict의 None 컬럼을 0.0으로 대체. weather 자체가 None이면 전체 0."""
    if weather is None:
        return {col: 0.0 for col in WEATHER_COLS}
    return {col: (float(weather[col]) if weather.get(col) is not None else 0.0) for col in WEATHER_COLS}


def _build_row(
    d: date,
    unit_id: int,
    weather: dict,
    calendar: dict,
    lag_map: dict[tuple[date, int], float],
) -> dict:
    """
    단일 날짜·unit_id 행의 feature 딕셔너리 생성.

    unit_id:
      - category 모델: category_id
      - cat×sub 모델:  category_id * 10_000 + subcategory_id  (복합 정수키)

    label encoding 없이 unit_id 정수를 그대로 feature로 사용.
    LightGBM 트리 계열은 값의 크기보다 분기점(split)만 학습하므로 문제 없음.
    """
    row = {
        "month":       d.month,
        "day":         d.day,
        "day_of_week": d.weekday(),
        "day_of_year": d.timetuple().tm_yday,
        "is_weekend":  int(calendar["is_weekend"]),
        "is_holiday":  int(calendar["is_holiday"]),
        "unit_id":     unit_id,
    }
    for col in WEATHER_COLS:
        row[col] = weather[col]
    for lag in LAG_DAYS:
        lag_date = d - timedelta(days=lag)
        # lag: 해당 날짜(point-in-time) 공고 수. 없으면 0.
        row[f"lag_{lag}d"] = lag_map.get((lag_date, unit_id), 0.0)

    return row



def collect_training_rows(
    origin_dates: list[date],
    unit_ids: list[int],
    actual_map: dict[tuple[date, int], float],
    weather_map: dict[date, dict],
    calendar_map: dict[date, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Auto-regressive 방식으로 학습 데이터 수집.

    각 origin date마다 그 시점 이전의 실제값만으로 lag_map을 초기화하고
    예측 window를 날짜 순서대로 순회하며 (feature, label) 쌍을 수집.
    날씨·캘린더 결측 날짜는 row를 추가하지 않되 lag_map은 업데이트.

    Args:
        origin_dates: 학습 기준일 목록 (순서 무관)
        unit_ids:     예측 대상 unit_id 전체 목록
        actual_map:   {(date, unit_id): count}
                      lag 소스(window 이전)와 학습 대상(window 내부) 모두 포함
        weather_map:  {date: weather_dict}
        calendar_map: {date: calendar_dict}

    Returns:
        X: (N, len(FEATURE_COLS))
        y: (N,)
    """
    sorted_entries = sorted(actual_map.items(), key=lambda kv: kv[0][0])

    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    total_skipped = 0
    pre_d_map: dict[tuple[date, int], float] = {}
    ptr = 0

    for d in sorted(origin_dates):
        # point-in-time lag_map: date < d 인 실제값만 포함
        while ptr < len(sorted_entries) and sorted_entries[ptr][0][0] < d:
            key, val = sorted_entries[ptr]
            pre_d_map[key] = val
            ptr += 1

        lag_map = dict(pre_d_map)
        pd = pred_end_date(d)
        pred_dates = [d + timedelta(days=i) for i in range((pd - d).days + 1)]

        for t in pred_dates:
            calendar = calendar_map.get(t)
            if calendar is None:
                logger.warning(f"[collect_training_rows] 캘린더 결측 — origin={d}, date={t}. skip.")
                total_skipped += 1
                for uid in unit_ids:
                    lag_map[(t, uid)] = actual_map.get((t, uid), 0.0)
                continue

            weather = _coerce_weather(weather_map.get(t))

            for uid in unit_ids:
                row = _build_row(t, uid, weather, calendar, lag_map)
                rows_X.append([row[col] for col in FEATURE_COLS])
                rows_y.append(actual_map.get((t, uid), 0.0))

            for uid in unit_ids:
                lag_map[(t, uid)] = actual_map.get((t, uid), 0.0)

    if total_skipped:
        logger.warning(f"[collect_training_rows] 총 {total_skipped}개 날짜 skip.")

    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def train(X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
    """LightGBM 학습"""
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X, y,
        feature_name=FEATURE_COLS,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    logger.info("[PredictionModel] Training complete.")
    return model


def save_models(
    weights_dir: str,
    cat_model: lgb.LGBMRegressor,
    cat_sub_model: lgb.LGBMRegressor,
    cat_unit_ids: list[int],
    cat_sub_unit_ids: list[int],
    train_days: int,
) -> None:
    """
    학습된 모델 가중치 및 메타데이터 저장.

    meta.json에 train_days를 함께 저장하여
    추론 시 lag 소스 범위를 학습과 동일하게 맞춤.

    cat_sub_unit_ids의 unit_id = category_id * 10_000 + subcategory_id
    """
    path = Path(weights_dir)
    path.mkdir(parents=True, exist_ok=True)

    cat_model.booster_.save_model(str(path / CATEGORY_MODEL_FILE))
    cat_sub_model.booster_.save_model(str(path / CAT_SUB_MODEL_FILE))

    meta = {
        "cat_unit_ids":     cat_unit_ids,
        "cat_sub_unit_ids": cat_sub_unit_ids,
        "train_days":       train_days,
    }
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(f"[PredictionModel] Saved models to {weights_dir}")


def load_models(
    weights_dir: str,
) -> tuple[lgb.Booster, lgb.Booster, list[int], list[int], int]:
    """
    저장된 모델 가중치 및 메타데이터 로드.

    Returns:
        cat_booster, cat_sub_booster, cat_unit_ids, cat_sub_unit_ids, train_days
    """
    path = Path(weights_dir)

    cat_booster     = lgb.Booster(model_file=str(path / CATEGORY_MODEL_FILE))
    cat_sub_booster = lgb.Booster(model_file=str(path / CAT_SUB_MODEL_FILE))

    meta            = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    cat_unit_ids:     list[int] = meta["cat_unit_ids"]
    cat_sub_unit_ids: list[int] = meta["cat_sub_unit_ids"]
    train_days:       int       = meta["train_days"]

    logger.info(f"[PredictionModel] Loaded models from {weights_dir} (train_days={train_days})")
    return cat_booster, cat_sub_booster, cat_unit_ids, cat_sub_unit_ids, train_days


def predict_one(
    booster: lgb.Booster,
    d: date,
    unit_id: int,
    weather: dict,
    calendar: dict,
    lag_map: dict[tuple[date, int], float],
) -> int:
    """
    단일 날짜·unit 예측 (lgb.Booster 사용).

    Returns:
        예측 공고 수 (음수 → 0, 반올림)
    """
    row = _build_row(d, unit_id, weather, calendar, lag_map)
    X = pd.DataFrame([[row[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    pred = booster.predict(X)[0]
    return max(0, round(float(pred)))


class PredictionModel:
    cat_booster: lgb.Booster | None = None
    cat_sub_booster: lgb.Booster | None = None
    cat_unit_ids: list[int] = []
    cat_sub_unit_ids: list[int] = []
    train_days: int = 0

    @classmethod
    def load(cls) -> None:
        (
            cls.cat_booster,
            cls.cat_sub_booster,
            cls.cat_unit_ids,
            cls.cat_sub_unit_ids,
            cls.train_days,
        ) = load_models(settings.model_weights_dir)

    @classmethod
    def predict_category(
        cls,
        d: date,
        unit_id: int,
        weather: dict,
        calendar: dict,
        lag_map: dict[tuple[date, int], float],
    ) -> int:
        return predict_one(cls.cat_booster, d, unit_id, weather, calendar, lag_map)

    @classmethod
    def predict_cat_sub(
        cls,
        d: date,
        unit_id: int,
        weather: dict,
        calendar: dict,
        lag_map: dict[tuple[date, int], float],
    ) -> int:
        return predict_one(cls.cat_sub_booster, d, unit_id, weather, calendar, lag_map)