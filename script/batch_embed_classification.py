import asyncio
from datetime import date

from app.service.classification_service import classify_all_unclassified

classify_all_unclassified(date(2024, 1, 1))