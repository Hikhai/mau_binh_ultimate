"""
FastAPI REST API for Mau Binh Solver
Monetize through API calls
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
from datetime import datetime
import hashlib
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from card import Deck
from ultimate_solver import UltimateSolver, SolverMode

app = FastAPI(
    title="Mau Binh Solver API",
    description="Professional Mau Binh AI Solver",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== MODELS ====================

class SolveRequest(BaseModel):
    cards: str  # Space-separated card string
    mode: str = "balanced"  # fast, balanced, accurate, ultimate
    api_key: Optional[str] = None


class SolveResponse(BaseModel):
    back: List[str]
    middle: List[str]
    front: List[str]
    ev: float
    bonus: int
    p_scoop: float
    computation_time: float
    credits_used: int


class UsageStats(BaseModel):
    total_requests: int
    remaining_credits: int
    plan: str


# ==================== DATABASE ====================

class APIKeyManager:
    """Manage API keys and usage tracking"""
    
    def __init__(self, db_path: str = "../../data/api_keys.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                plan TEXT,
                credits_remaining INTEGER,
                total_requests INTEGER,
                created_at TEXT,
                last_used TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_key TEXT,
                timestamp TEXT,
                mode TEXT,
                credits_used INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_key(self, plan: str = "free") -> str:
        """Create new API key"""
        # Generate key
        key = hashlib.sha256(f"{datetime.now()}{plan}".encode()).hexdigest()[:32]
        
        # Credits based on plan
        credits = {
            "free": 100,
            "basic": 1000,
            "pro": 10000,
            "unlimited": 999999
        }
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO api_keys (key, plan, credits_remaining, total_requests, created_at, last_used)
            VALUES (?, ?, ?, 0, ?, NULL)
        ''', (key, plan, credits.get(plan, 100), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        return key
    
    def validate_key(self, key: str, credits_needed: int) -> bool:
        """Validate API key and check credits"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT credits_remaining FROM api_keys WHERE key = ?', (key,))
        result = c.fetchone()
        
        conn.close()
        
        if not result:
            return False
        
        return result[0] >= credits_needed
    
    def use_credits(self, key: str, credits: int, mode: str):
        """Deduct credits"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE api_keys
            SET credits_remaining = credits_remaining - ?,
                total_requests = total_requests + 1,
                last_used = ?
            WHERE key = ?
        ''', (credits, datetime.now().isoformat(), key))
        
        c.execute('''
            INSERT INTO requests (api_key, timestamp, mode, credits_used)
            VALUES (?, ?, ?, ?)
        ''', (key, datetime.now().isoformat(), mode, credits))
        
        conn.commit()
        conn.close()
    
    def get_stats(self, key: str) -> dict:
        """Get usage statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT plan, credits_remaining, total_requests
            FROM api_keys
            WHERE key = ?
        ''', (key,))
        
        result = c.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return {
            'plan': result[0],
            'remaining_credits': result[1],
            'total_requests': result[2]
        }


# ==================== API ENDPOINTS ====================

api_key_manager = APIKeyManager()


def get_api_key(x_api_key: str = Header(None)) -> str:
    """Dependency to validate API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    return x_api_key


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest, api_key: str = Depends(get_api_key)):
    """
    Solve Mau Binh hand
    
    **Pricing:**
    - Fast mode: 1 credit
    - Balanced mode: 3 credits
    - Accurate mode: 5 credits
    - Ultimate mode: 10 credits
    """
    # Credit costs
    credit_costs = {
        "fast": 1,
        "balanced": 3,
        "accurate": 5,
        "ultimate": 10
    }
    
    mode = request.mode.lower()
    credits_needed = credit_costs.get(mode, 3)
    
    # Validate API key
    if not api_key_manager.validate_key(api_key, credits_needed):
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    try:
        # Parse cards
        cards = Deck.parse_hand(request.cards)
        
        if len(cards) != 13:
            raise HTTPException(status_code=400, detail="Need exactly 13 cards")
        
        # Solve
        solver_mode = SolverMode(mode)
        solver = UltimateSolver(cards, mode=solver_mode, verbose=False)
        result = solver.solve()
        
        # Deduct credits
        api_key_manager.use_credits(api_key, credits_needed, mode)
        
        # Return response
        return SolveResponse(
            back=[str(c) for c in result.back],
            middle=[str(c) for c in result.middle],
            front=[str(c) for c in result.front],
            ev=result.ev,
            bonus=result.bonus,
            p_scoop=result.p_scoop,
            computation_time=result.computation_time,
            credits_used=credits_needed
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=UsageStats)
async def get_stats(api_key: str = Depends(get_api_key)):
    """Get usage statistics"""
    stats = api_key_manager.get_stats(api_key)
    
    if not stats:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return UsageStats(
        total_requests=stats['total_requests'],
        remaining_credits=stats['remaining_credits'],
        plan=stats['plan']
    )


@app.get("/pricing")
async def pricing():
    """Get pricing information"""
    return {
        "plans": {
            "free": {
                "price": "$0",
                "credits": 100,
                "features": ["Basic solver", "Fast mode only"]
            },
            "basic": {
                "price": "$9.99/month",
                "credits": 1000,
                "features": ["All modes", "Priority support"]
            },
            "pro": {
                "price": "$49.99/month",
                "credits": 10000,
                "features": ["All modes", "API access", "Advanced features"]
            },
            "unlimited": {
                "price": "$199.99/month",
                "credits": "Unlimited",
                "features": ["Everything", "Dedicated support", "Custom solutions"]
            }
        },
        "credit_costs": {
            "fast": 1,
            "balanced": 3,
            "accurate": 5,
            "ultimate": 10
        }
    }


@app.post("/demo")
async def demo(request: SolveRequest):
    """Demo endpoint (no API key required, limited to fast mode)"""
    try:
        cards = Deck.parse_hand(request.cards)
        
        solver = UltimateSolver(cards, mode=SolverMode.FAST, verbose=False)
        result = solver.solve()
        
        return {
            "back": [str(c) for c in result.back],
            "middle": [str(c) for c in result.middle],
            "front": [str(c) for c in result.front],
            "ev": result.ev,
            "note": "Demo mode - Sign up for full features!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ADMIN ====================

@app.post("/admin/create-key")
async def create_key(plan: str = "free", admin_password: str = Header(None)):
    """Create new API key (admin only)"""
    if admin_password != "your_secret_admin_password":  # Change this!
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    key = api_key_manager.create_key(plan)
    
    return {"api_key": key, "plan": plan}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Mau Binh Solver API Server")
    print("📖 Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)