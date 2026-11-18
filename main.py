"""
Mr.AD backend — FastAPI app.

This is a real, runnable starting point, not a sketch: every route below
talks to the database and does the actual work (credit deduction, wallet
expiry, Claude calls). What it deliberately does NOT include yet:
  - Real payment gateway integration (the webhook route is stubbed with
    the signature-verification step clearly marked TODO — SSLCommerz/
    AamarPay/ShurjoPay each has its own verification scheme, and faking
    one here would be worse than leaving it honest)
  - Google/Facebook OAuth
  - Rate limiting / abuse protection (phone verification for free-tier
    farming, flagged back when this was first discussed)

Run locally:
    pip install -r requirements.txt
    cp .env.example .env   # fill in real values
    uvicorn main:app --reload
"""

import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from auth import create_token, generate_referral_code, hash_password, verify_password, verify_token
from chat import run_chat_turn
from wallet import (
    Batch,
    add_topup,
    active_balance,
    cost_for_message,
    current_year_month,
    deduct,
    free_questions_left,
    sweep_expired,
)

DATABASE_URL = os.environ["DATABASE_URL"]

app = FastAPI(title="Mr.AD API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    mobile: str
    name: str
    password: str
    referral_code: str | None = None


class LoginRequest(BaseModel):
    mobile: str
    password: str


@app.post("/auth/signup")
def signup(req: SignupRequest):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id FROM users WHERE mobile = %s", (req.mobile,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="An account with this mobile number already exists")

        referred_by = None
        if req.referral_code:
            cur.execute("SELECT id FROM users WHERE referral_code = %s", (req.referral_code,))
            row = cur.fetchone()
            if row:
                referred_by = row["id"]

        my_code = generate_referral_code()
        cur.execute(
            """
            INSERT INTO users (mobile, name, password_hash, referral_code, referred_by)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """,
            (req.mobile, req.name, hash_password(req.password), my_code, referred_by),
        )
        user_id = cur.fetchone()["id"]

        # Referral bonus — same 50-credit amount already shown to users in
        # the prototype's "Refer & earn" modal. Unlike the client-only
        # prototype (which could only credit the NEW signer, never the
        # referrer, for lack of a server), this can finally do both sides.
        if referred_by:
            now = datetime.now(timezone.utc)
            for uid in (user_id, referred_by):
                cur.execute(
                    "INSERT INTO wallet_batches (user_id, credits, purchased_at, expires_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (uid, 50, now, now + timedelta(days=30)),
                )

        token = create_token(str(user_id))
        return {"token": token, "user_id": str(user_id), "referral_code": my_code}


class GoogleAuthRequest(BaseModel):
    credential: str
    referral_code: str | None = None


@app.post("/auth/google")
def google_login(req: GoogleAuthRequest):
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token

        info = id_token.verify_oauth2_token(req.credential, google_requests.Request(), client_id)
    except Exception:
        raise HTTPException(status_code=401, detail="Google sign-in could not be verified")
    email = info.get("email")
    if not email or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google account email is not verified")
    email = email.lower()
    name = info.get("name") or email.split("@")[0]

    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, referral_code FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        if row:
            return {"token": create_token(str(row["id"])), "user_id": str(row["id"]),
                    "referral_code": row["referral_code"]}

        referred_by = None
        if req.referral_code:
            cur.execute("SELECT id FROM users WHERE referral_code = %s", (req.referral_code,))
            ref = cur.fetchone()
            if ref:
                referred_by = ref["id"]

        my_code = generate_referral_code()
        cur.execute(
            """
            INSERT INTO users (name, email, referral_code, referred_by)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (name, email, my_code, referred_by),
        )
        user_id = cur.fetchone()["id"]

        if referred_by:
            now = datetime.now(timezone.utc)
            for uid in (user_id, referred_by):
                cur.execute(
                    "INSERT INTO wallet_batches (user_id, credits, purchased_at, expires_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (uid, 50, now, now + timedelta(days=30)),
                )
        return {"token": create_token(str(user_id)), "user_id": str(user_id), "referral_code": my_code}


@app.post("/auth/login")
def login(req: LoginRequest):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, password_hash FROM users WHERE mobile = %s", (req.mobile,))
        row = cur.fetchone()
        if not row or not row["password_hash"] or not verify_password(req.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid mobile number or password")
        return {"token": create_token(str(row["id"]))}


# ---------------------------------------------------------------------------
# Wallet
# ---------------------------------------------------------------------------

def _load_batches(cur, user_id: str) -> list[Batch]:
    cur.execute(
        "SELECT id, credits, purchased_at, expires_at, expired FROM wallet_batches WHERE user_id = %s",
        (user_id,),
    )
    return [Batch(id=str(r["id"]), credits=r["credits"], purchased_at=r["purchased_at"],
                   expires_at=r["expires_at"], expired=r["expired"]) for r in cur.fetchall()]


@app.get("/wallet/balance")
def wallet_balance(user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        batches = _load_batches(cur, user_id)
        batches, newly_expired = sweep_expired(batches)
        for b in batches:
            cur.execute("UPDATE wallet_batches SET expired = %s WHERE id = %s", (b.expired, b.id))

        cur.execute("SELECT used FROM free_quota WHERE user_id = %s AND year_month = %s",
                    (user_id, current_year_month()))
        row = cur.fetchone()
        used = row["used"] if row else 0

        return {
            "credit_balance": active_balance(batches),
            "free_questions_left": free_questions_left(used),
        }


# ---------------------------------------------------------------------------
# Payment webhook — gateway-agnostic shape, signature check is the one part
# left for whichever gateway (SSLCommerz/AamarPay/ShurjoPay) is actually
# chosen, since each has its own verification scheme.
# ---------------------------------------------------------------------------

class PaymentWebhook(BaseModel):
    user_id: str
    gateway_txn_id: str
    gateway_name: str
    amount_bdt: float
    credits: int
    signature: str  # TODO: verify against the gateway's actual signing scheme


@app.post("/webhook/payment")
def payment_webhook(payload: PaymentWebhook):
    raise HTTPException(status_code=503, detail="Payment webhook disabled until signature check is added")
    # TODO before going live: verify `payload.signature` against the chosen
    # gateway's documented scheme. Until this is real, this endpoint must
    # NOT be exposed publicly — a forged request would mint free credits.
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Idempotency: gateway_txn_id is UNIQUE, so a retried webhook is a
        # no-op rather than double-crediting the same payment.
        cur.execute(
            "SELECT id FROM payment_transactions WHERE gateway_txn_id = %s",
            (payload.gateway_txn_id,),
        )
        if cur.fetchone():
            return {"status": "already_processed"}

        cur.execute(
            """
            INSERT INTO payment_transactions
                (user_id, gateway_txn_id, gateway_name, amount_bdt, credits, status)
            VALUES (%s, %s, %s, %s, %s, 'success')
            """,
            (payload.user_id, payload.gateway_txn_id, payload.gateway_name,
             payload.amount_bdt, payload.credits),
        )

        batches = _load_batches(cur, payload.user_id)
        batches, newly_expired_now = sweep_expired(batches)
        # expired_pool would normally be tracked on the user row across
        # calls; simplified here to "credits that just expired this check."
        result = add_topup(batches, newly_expired_now, payload.credits)

        now = datetime.now(timezone.utc)
        new_batch = result["batches"][-1]
        cur.execute(
            "INSERT INTO wallet_batches (user_id, credits, purchased_at, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (payload.user_id, new_batch.credits, now, now + timedelta(days=30)),
        )

        return {"status": "success", "reactivated_credits": result["reactivated"]}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

class CreateCaseRequest(BaseModel):
    mode: str = "plain"
    lang: str = "en"


@app.post("/cases")
def create_case(req: CreateCaseRequest, user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "INSERT INTO cases (user_id, mode, lang) VALUES (%s, %s, %s) RETURNING id",
            (user_id, req.mode, req.lang),
        )
        return {"case_id": str(cur.fetchone()["id"])}


@app.get("/cases")
def list_cases(user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, mode, lang, created_at FROM cases WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        )
        return {"cases": cur.fetchall()}


# ---------------------------------------------------------------------------
# Chat — the core endpoint. Checks quota/credits BEFORE calling Claude, so a
# failed or declined request never costs the user anything.
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    case_id: str
    message: str
    has_image: bool = False


@app.post("/chat")
def chat(req: ChatRequest, user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT user_id, mode, lang FROM cases WHERE id = %s", (req.case_id,))
        case = cur.fetchone()
        if not case or str(case["user_id"]) != user_id:
            raise HTTPException(status_code=404, detail="Case not found")

        cost = cost_for_message(case["mode"], req.has_image)

        # Free quota first, then wallet — same order as the prototype.
        ym = current_year_month()
        cur.execute("SELECT used FROM free_quota WHERE user_id = %s AND year_month = %s", (user_id, ym))
        quota_row = cur.fetchone()
        used = quota_row["used"] if quota_row else 0
        use_free = free_questions_left(used) > 0

        if not use_free:
            batches = _load_batches(cur, user_id)
            batches, _ = sweep_expired(batches)
            if active_balance(batches) < cost:
                raise HTTPException(status_code=402, detail="Insufficient credits — top up to continue")

        # Load conversation history for this case.
        cur.execute(
            "SELECT role, text FROM messages WHERE case_id = %s ORDER BY created_at ASC",
            (req.case_id,),
        )
        history = [{"role": r["role"], "content": r["text"]} for r in cur.fetchall()]
        history.append({"role": "user", "content": req.message})

        cur.execute(
            "INSERT INTO messages (case_id, role, text) VALUES (%s, 'user', %s)",
            (req.case_id, req.message),
        )

        result = run_chat_turn(conn, history, case["lang"], case["mode"])

        # Only charge AFTER a successful response.
        if use_free:
            if quota_row:
                cur.execute(
                    "UPDATE free_quota SET used = used + 1 WHERE user_id = %s AND year_month = %s",
                    (user_id, ym),
                )
            else:
                cur.execute(
                    "INSERT INTO free_quota (user_id, year_month, used) VALUES (%s, %s, 1)",
                    (user_id, ym),
                )
        else:
            batches = deduct(batches, cost)
            for b in batches:
                cur.execute("UPDATE wallet_batches SET credits = %s WHERE id = %s", (b.credits, b.id))

        cur.execute(
            "INSERT INTO messages (case_id, role, text, sources_json, cost_credits) "
            "VALUES (%s, 'assistant', %s, %s, %s) RETURNING id",
            (req.case_id, result["text"], psycopg2.extras.Json(result["sources"]),
             0 if use_free else cost),
        )
        message_id = cur.fetchone()["id"]

        return {"text": result["text"], "sources": result["sources"], "message_id": str(message_id)}


# ---------------------------------------------------------------------------
# Feedback & lawyer requests
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    message_id: str
    rating: str  # 'up' | 'down'
    mode: str
    lang: str


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest, user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (message_id, user_id, rating, mode, lang) VALUES (%s, %s, %s, %s, %s)",
            (req.message_id, user_id, req.rating, req.mode, req.lang),
        )
    return {"status": "recorded"}


class LawyerRequestPayload(BaseModel):
    name: str
    mobile: str
    brief: str | None = None


@app.post("/lawyer-requests")
def submit_lawyer_request(req: LawyerRequestPayload, user_id: str = Depends(verify_token)):
    with get_db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO lawyer_requests (user_id, name, mobile, brief) VALUES (%s, %s, %s, %s)",
            (user_id, req.name, req.mobile, req.brief),
        )
    return {"status": "recorded"}
