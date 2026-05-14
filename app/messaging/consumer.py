import logging
import json
import asyncio
from datetime import date
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.service.keyword_classification_service import classify_one_day

logger = logging.getLogger(__name__)

_consumer: AIOKafkaConsumer | None = None

async def start_consumer() -> None:
    """
    Kafka 컨슈머를 시작하고 메시지 수신 루프를 실행
    FastAPI lifespan startup 시 호출됨
    """
    
    global _consumer
    
    _consumer = AIOKafkaConsumer(
        settings.kafka_topic_announcement_created,
        settings.kafka_topic_prediction_request,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group_id,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: v.decode("utf-8"),
    )
    
    await _consumer.start()
    logger.info(
        f"Kafka 컨슈머 시작: topic={settings.kafka_topic_announcement_created} 수신, "
        f"group={settings.kafka_consumer_group_id}"
    )
    
    async for message in _consumer:
        await _handle_message(message)

async def stop_consumer() -> None:
    """
    Kafka 컨슈머를 정상 종료
    FastAPI lifespan shutdown 시 호출됨
    """
    
    global _consumer
    
    if _consumer:
        await _consumer.stop()
        logger.info("Kafka 컨슈머 종료 완료")
        _consumer = None

async def _handle_message(message) -> None:
    try:
        print(f"메시지 수신: {message}")
        date_str = message.value
        d = date.fromisoformat(date_str)
        
        await asyncio.to_thread(classify_one_day, d)
        logger.info(f"메시지 처리 완료: from {d}")
    except Exception as e:
        logger.error(f"시그널 처리 실패: {e}", exc_info=True)