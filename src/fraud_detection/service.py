"""FastAPI health and scoring endpoints for transaction risk evaluation."""

from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Transaction payload accepted by the scoring endpoint."""

    amount: float = Field(..., gt=0)
    type: str = "PAYMENT"
    nameOrig: str | None = None
    nameDest: str | None = None
    oldbalanceOrg: float | None = None
    newbalanceDest: float | None = None


def _sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""

    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def score_transaction(payload: dict[str, Any] | TransactionRequest) -> float:
    """Return a fraud risk score between 0 and 1 for a transaction payload."""

    if isinstance(payload, TransactionRequest):
        amount = float(payload.amount)
        transaction_type = str(payload.type).upper()
        name_orig = payload.nameOrig
        name_dest = payload.nameDest
    else:
        amount = float(payload.get("amount", 0.0))
        transaction_type = str(payload.get("type", "PAYMENT")).upper()
        name_orig = payload.get("nameOrig")
        name_dest = payload.get("nameDest")

    if amount <= 0:
        return 0.0

    raw_score = math.log1p(amount) * 0.45
    if transaction_type in {"TRANSFER", "CASH_OUT", "CASH_IN"}:
        raw_score += 0.60
    elif transaction_type == "PAYMENT":
        raw_score += 0.10
    else:
        raw_score += 0.20

    if name_orig and name_dest:
        raw_score += 0.05

    raw_score -= 2.2
    return float(_sigmoid(raw_score))


def create_app() -> FastAPI:
    """Create the fraud detection FastAPI application."""

    app = FastAPI(title="Real-time Fraud Detection API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score")
    def score(request: TransactionRequest) -> dict[str, Any]:
        probability = score_transaction(request)
        return {
            "status": "ok",
            "score": probability,
            "type": request.type,
            "amount": request.amount,
        }

    return app


app = create_app()
