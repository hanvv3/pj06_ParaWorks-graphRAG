import time
from collections import defaultdict

from fastapi import HTTPException, Request

# Simple in-memory storage for rate limiting
# In production, this should be moved to Redis
_LIMITER_STORAGE = defaultdict(list)

def is_rate_limited(key: str, limit: int, window_seconds: int) -> bool:
    now = time.time()
    # Clean up old entries
    _LIMITER_STORAGE[key] = [t for t in _LIMITER_STORAGE[key] if t > now - window_seconds]
    
    if len(_LIMITER_STORAGE[key]) >= limit:
        return True
    
    _LIMITER_STORAGE[key].append(now)
    return False

async def rate_limit_auth(request: Request):
    # We use the client IP as the key
    client_ip = request.client.host if request.client else "unknown"
    
    # 5 login/refresh attempts per minute per IP
    if is_rate_limited(f"auth:{client_ip}", limit=5, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many authentication attempts. Please try again later.")
