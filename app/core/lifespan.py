import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.model.kobert import KoBERTModel
from app.messaging.consumer import start_consumer, stop_consumer
from app.messaging.producer import start_producer, stop_producer

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 시작/종료 처리

    yield 이전: startup (앱 시작 시 1회 실행)
    yield 이후: shutdown (앱 종료 시 1회 실행)
    """
    
    logger.info("앱 시작 중...")
    
    KoBERTModel.load()
    logger.info("KoBERT 모델 로드 완료")
    
    await start_producer()
    
    consumer_task = asyncio.create_task(start_consumer())
    logger.info("Kafka 컨슈머 백그라운드 태스크 등록 완료")

    logger.info("앱 시작 완료")
    
    yield
    
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    
    await stop_consumer()
    await stop_producer()
    
    logger.info("앱 종료 완료")