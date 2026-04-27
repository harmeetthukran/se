"""Upstox live broker adapter — places REAL orders.

This is the production order-placement path. Every public method that
mutates account state requires `live=True` AND `UPSTOX_LIVE_OK=1` in the
environment. Default is dry-run: orders are logged and dropped.

Endpoints used (Upstox v2):
    POST   /v2/order/place
    POST   /v2/order/cancel
    GET    /v2/order/details
    GET    /v2/portfolio/short-term-positions
    GET    /v2/portfolio/long-term-holdings
    GET    /v2/user/profile (used as a sanity ping)

NEVER call this module from a strategy directly — call paper.py first,
prove the strategy out for at least 4-8 weeks, THEN flip a single
runner script to use this. Treat live trading as "paper mode with
consequences": the API surface mirrors paper.py.

Safety guards:
    * Dry-run by default. Even with live=True, requires the env var
      UPSTOX_LIVE_OK=1 to be set in the current process. Two locks.
    * Per-day notional cap (`max_notional_per_day`). When exceeded,
      further place_order() calls return an error stub.
    * Per-day order-count cap (`max_orders_per_day`). Sanity bound
      against a runaway loop hammering orders.
    * Dedupe by `client_order_id` so a retry doesn't double-fill.
    * All order attempts logged to data/live/orders_audit.parquet.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd

from nse_bot.config import DATA_DIR

LIVE_DIR = DATA_DIR / "live"
AUDIT_PATH = LIVE_DIR / "orders_audit.parquet"

BASE_URL = "https://api.upstox.com/v2"

OrderSide = Literal["BUY", "SELL"]
TransactionType = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT", "SL", "SL-M"]
ProductType = Literal["I", "D", "CO", "MTF"]  # I=intraday MIS, D=delivery CNC


class BrokerError(RuntimeError):
    pass


class NotPermittedError(BrokerError):
    pass


@dataclass
class Order:
    instrument_token: str
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType = "MARKET"
    product: ProductType = "I"
    price: float = 0.0
    trigger_price: float = 0.0
    validity: str = "DAY"
    disclosed_quantity: int = 0
    is_amo: bool = False
    tag: str = ""

    def to_payload(self, client_order_id: str) -> dict[str, Any]:
        return {
            "quantity": int(self.quantity),
            "product": self.product,
            "validity": self.validity,
            "price": float(self.price),
            "tag": self.tag or "",
            "instrument_token": self.instrument_token,
            "order_type": self.order_type,
            "transaction_type": self.transaction_type,
            "disclosed_quantity": int(self.disclosed_quantity),
            "trigger_price": float(self.trigger_price),
            "is_amo": bool(self.is_amo),
            "correlation_id": client_order_id,
        }


@dataclass
class BrokerLimits:
    max_orders_per_day: int = 50
    max_notional_per_day: float = 500_000.0  # ₹5L per day default cap


@dataclass
class _DailyState:
    day: date
    order_count: int = 0
    notional_total: float = 0.0
    seen_correlation_ids: set[str] = field(default_factory=set)


class UpstoxBroker:
    """Thin wrapper. live=False (default) is dry-run.

    Even with live=True the module also requires env UPSTOX_LIVE_OK=1.
    """

    def __init__(
        self,
        access_token: str,
        *,
        live: bool = False,
        limits: BrokerLimits | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not access_token:
            raise BrokerError("Missing access_token. Run scripts/auth.py first.")
        self._token = access_token
        self._live = bool(live)
        self._limits = limits or BrokerLimits()
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        self._state = _DailyState(day=date.today())
        LIVE_DIR.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "UpstoxBroker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- read-only ----

    def ping(self) -> dict:
        r = self._client.get("/user/profile")
        if r.status_code >= 400:
            raise BrokerError(f"profile {r.status_code}: {r.text[:200]}")
        return r.json()

    def positions(self) -> pd.DataFrame:
        r = self._client.get("/portfolio/short-term-positions")
        if r.status_code >= 400:
            raise BrokerError(f"positions {r.status_code}: {r.text[:200]}")
        data = r.json().get("data") or []
        return pd.DataFrame(data)

    def holdings(self) -> pd.DataFrame:
        r = self._client.get("/portfolio/long-term-holdings")
        if r.status_code >= 400:
            raise BrokerError(f"holdings {r.status_code}: {r.text[:200]}")
        data = r.json().get("data") or []
        return pd.DataFrame(data)

    def order_details(self, order_id: str) -> dict:
        r = self._client.get("/order/details", params={"order_id": order_id})
        if r.status_code >= 400:
            raise BrokerError(f"order_details {r.status_code}: {r.text[:200]}")
        return r.json()

    # ---- mutating ----

    def place_order(
        self,
        order: Order,
        *,
        client_order_id: str | None = None,
        approx_notional: float | None = None,
    ) -> dict:
        """Place an order. In dry-run, logs and returns a stub.

        Returns the broker response dict (or stub). Always also persisted
        to data/live/orders_audit.parquet.
        """
        if self._state.day != date.today():
            self._state = _DailyState(day=date.today())

        client_order_id = client_order_id or str(uuid.uuid4())
        if client_order_id in self._state.seen_correlation_ids:
            return {"status": "duplicate", "correlation_id": client_order_id}

        if self._state.order_count >= self._limits.max_orders_per_day:
            self._audit(order, client_order_id, status="rejected_daily_count_cap")
            raise BrokerError(
                f"max_orders_per_day cap ({self._limits.max_orders_per_day}) reached"
            )

        notional_est = float(approx_notional or order.quantity * max(order.price, 1.0))
        if self._state.notional_total + notional_est > self._limits.max_notional_per_day:
            self._audit(order, client_order_id, status="rejected_daily_notional_cap")
            raise BrokerError(
                f"daily notional cap (₹{self._limits.max_notional_per_day:,.0f}) would be exceeded"
            )

        if not self._live or os.getenv("UPSTOX_LIVE_OK", "").strip() != "1":
            self._audit(order, client_order_id, status="dry_run")
            self._state.order_count += 1
            self._state.notional_total += notional_est
            self._state.seen_correlation_ids.add(client_order_id)
            return {"status": "dry_run", "correlation_id": client_order_id}

        payload = order.to_payload(client_order_id)
        r = self._client.post("/order/place", json=payload)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        if r.status_code >= 400:
            self._audit(order, client_order_id, status=f"http_{r.status_code}", response=body)
            raise BrokerError(f"place_order {r.status_code}: {r.text[:300]}")

        self._state.order_count += 1
        self._state.notional_total += notional_est
        self._state.seen_correlation_ids.add(client_order_id)
        self._audit(order, client_order_id, status="placed", response=body)
        return body

    def cancel_order(self, order_id: str) -> dict:
        if not self._live or os.getenv("UPSTOX_LIVE_OK", "").strip() != "1":
            return {"status": "dry_run", "order_id": order_id}
        r = self._client.delete("/order/cancel", params={"order_id": order_id})
        if r.status_code >= 400:
            raise BrokerError(f"cancel {r.status_code}: {r.text[:200]}")
        return r.json()

    def kill_switch(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        if not self._live or os.getenv("UPSTOX_LIVE_OK", "").strip() != "1":
            return 0
        r = self._client.get("/order/retrieve-all")
        if r.status_code >= 400:
            return 0
        items = r.json().get("data") or []
        cancelled = 0
        for it in items:
            status = (it.get("status") or "").lower()
            if status in ("complete", "cancelled", "rejected"):
                continue
            try:
                self.cancel_order(it.get("order_id") or "")
                cancelled += 1
            except BrokerError:
                continue
        return cancelled

    # ---- audit ----

    def _audit(
        self,
        order: Order,
        correlation_id: str,
        *,
        status: str,
        response: dict | None = None,
    ) -> None:
        row = {
            "ts": datetime.now(),
            "correlation_id": correlation_id,
            "live": bool(self._live and os.getenv("UPSTOX_LIVE_OK", "").strip() == "1"),
            "status": status,
            "instrument_token": order.instrument_token,
            "transaction_type": order.transaction_type,
            "quantity": int(order.quantity),
            "order_type": order.order_type,
            "product": order.product,
            "price": float(order.price),
            "tag": order.tag,
            "response": str(response)[:1000] if response is not None else "",
        }
        if AUDIT_PATH.exists():
            existing = pd.read_parquet(AUDIT_PATH)
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            existing = pd.DataFrame([row])
        existing.to_parquet(AUDIT_PATH, index=False)
