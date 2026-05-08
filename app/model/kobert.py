import logging
import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from app.core.config import settings

logger = logging.getLogger(__name__)

class KoBERTModel:
    _tokenizer = None
    _model = None
    
    @classmethod
    def load(cls):
        """
        KoBERT 모델을 HuggingFace에서 다운로드 및 메모리에 로드
        lifespan.py의 startup에서 최초 1회만 호출
        이후 호출 시 이미 로드됐으면 스킵
        """

        if cls._model is not None:
            logger.info("KoBERT 모델이 이미 로드되어 있음")
            return
        
        logger.info("KoBERT 모델 로딩 중")
        cls._tokenizer = AutoTokenizer.from_pretrained(settings.kobert_model_name, trust_remote_code=True)
        cls._model = AutoModel.from_pretrained(settings.kobert_model_name, trust_remote_code=True)
        cls._model.eval()
        logger.info("KoBERT 모델 로드 완료")
    
    @classmethod
    def get(cls):
        """
        로드된 토크나이저와 모델 반환
        로드 안 됐으면 자동으로 로드 후 반환
        """
        
        if cls._model is None:
            cls.load()
        
        return cls._tokenizer, cls._model