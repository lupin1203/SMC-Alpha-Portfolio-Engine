# utils.py
import logging
import time
import requests
from typing import Any

# 設定日誌
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def safe_divide(a: float, b: float) -> float:
    """安全除法：若分母為 0 或出錯，回傳 0。"""
    try:
        return a / b
    except Exception:
        return 0.0


def fetch_with_retry(url: str, params: dict = None, retries: int = 3, timeout: int = 5) -> Any:
    """HTTP GET 請求，包含重試機制與逾時設定。"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            logger.warning(f"第 {attempt} 次 GET 請求失敗: {e}")
            time.sleep(1)
    logger.error(f"GET 請求連續 {retries} 次失敗: {url}")
    return None


def rate_limit(calls: int, period: float):
    """裝飾器：限制函數每 period 秒只允許 calls 次呼叫。"""
    def decorator(func):
        last_called = [0.0]
        call_count = [0]
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_called[0] > period:
                last_called[0] = now
                call_count[0] = 0
            if call_count[0] >= calls:
                sleep = period - (now - last_called[0])
                if sleep > 0:
                    time.sleep(sleep)
                last_called[0] = time.time()
                call_count[0] = 0
            call_count[0] += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator