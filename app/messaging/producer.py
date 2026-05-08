import logging
import json
from aiokafka import AIOKafkaProducer
from app.core.config import settings

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None

async def start_producer() -> None:
    """
    Kafka 프로듀서를 시작
    FastAPI lifespan startup 시 호출됨
    """
    
    global _producer
    
    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        acks="all"
    )
    
    await _producer.start()
    logger.info("Kafka 프로듀서 시작 완료")

async def stop_producer() -> None:
    """
    Kafka 프로듀서를 정상 종료 (버퍼에 남은 메시지 flush 후 종료)
    FastAPI lifespan shutdown 시 호출됨
    """
    global _producer
    if _producer:
        await _producer.stop()
        logger.info("Kafka 프로듀서 종료 완료")
        _producer = None

async def publish_prediction_response() -> None:
    """
    예측 완료 토픽 발행
    """
    
    if _producer is None:
        logger.error("프로듀서가 초기화되지 않음")
        return

    await _producer.send_and_wait(
        settings.kafka_topic_prediction_response,
        value={
            "event": "prediction-response"
        }
    )
    
    logger.info("예측 결과 발행 완료")