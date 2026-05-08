import logging.config
from fastapi import FastAPI
from app.core.lifespan import lifespan

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

app = FastAPI(
    title="공고 분류 및 예측 서비스",
    description="KoBERT + pgvector 기반 공고 분류 및 자체 모델 기반 공고 수 예측",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["health"])
async def health_check():
    """
    로드밸런서/쿠버네티스 헬스체크용
    """
    return {"status": "ok"}