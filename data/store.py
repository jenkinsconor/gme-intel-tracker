"""
SQLite persistence — IV history (IVR/IVP) and Wheel trade ledger.
Stored in project root as gme_intel.db — survives Streamlit restarts.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gme_intel.db")


class IVStore:
    """
    Persists daily constant-maturity 30-day IV snapshots.
    Accumulates history for IVR and IVP computation.

    IVR (IV Rank):       (current - 52w_low) / (52w_high - 52w_low) × 100
    IVP (IV Percentile): % of past 252 days where IV was below current
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iv_history (
                    date        TEXT PRIMARY KEY,
                    iv_30d      REAL NOT NULL,
                    spot        REAL NOT NULL,
                    recorded_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def upsert(self, iv_30d: float, spot: float, record_date: Optional[date] = None) -> None:
        """Store today's IV snapshot (one record per date — idempotent)."""
        d = record_date or date.today()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO iv_history (date, iv_30d, spot, recorded_at) VALUES (?, ?, ?, ?)",
                (d.isoformat(), iv_30d, spot, datetime.now().isoformat()),
            )
            conn.commit()

    def get_history(self, days: int = 365) -> pd.DataFrame:
        """Return IV history DataFrame for the past N calendar days."""
        since = (date.today() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, iv_30d, spot FROM iv_history WHERE date >= ? ORDER BY date",
                conn,
                params=(since,),
            )
        return df

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM iv_history").fetchone()[0]

    def latest(self) -> Optional[tuple[date, float]]:
        """Return (date, iv_30d) of the most recent record, or None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT date, iv_30d FROM iv_history ORDER BY date DESC LIMIT 1"
            ).fetchone()
        if row:
            return date.fromisoformat(row[0]), row[1]
        return None


# ---------------------------------------------------------------------------
# Wheel trade ledger
# ---------------------------------------------------------------------------

class WheelStore:
    """
    Persists Wheel cycles and legs in SQLite.
    Provides the cost-basis ledger — the source of truth for adjusted_basis_per_share.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wheel_cycles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT NOT NULL,
                    started_date TEXT NOT NULL,
                    closed_date  TEXT,
                    state        TEXT NOT NULL DEFAULT 'CSP_OPEN',
                    notes        TEXT DEFAULT '',
                    created_at   TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS wheel_legs (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id            INTEGER NOT NULL REFERENCES wheel_cycles(id),
                    leg_date            TEXT NOT NULL,
                    leg_type            TEXT NOT NULL,
                    strike              REAL NOT NULL,
                    premium_per_share   REAL NOT NULL DEFAULT 0,
                    contracts           INTEGER NOT NULL DEFAULT 1,
                    expiry              TEXT,
                    notes               TEXT DEFAULT '',
                    created_at          TEXT NOT NULL
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Cycle management
    # ------------------------------------------------------------------

    def open_cycle(self, ticker: str, notes: str = "") -> int:
        """Start a new Wheel cycle. Returns the new cycle_id."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO wheel_cycles (ticker, started_date, state, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (ticker, date.today().isoformat(), "CSP_OPEN", notes, datetime.now().isoformat()),
            )
            conn.commit()
            return cur.lastrowid

    def update_cycle_state(self, cycle_id: int, state: str, closed: bool = False) -> None:
        closed_date = date.today().isoformat() if closed else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE wheel_cycles SET state = ?, closed_date = ? WHERE id = ?",
                (state, closed_date, cycle_id),
            )
            conn.commit()

    def get_active_cycle(self, ticker: str) -> Optional[dict]:
        """Return the most recent non-terminal cycle for ticker, or None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT id, ticker, started_date, closed_date, state, notes
                FROM wheel_cycles
                WHERE ticker = ? AND state NOT IN ('CALLED_AWAY', 'CYCLE_COMPLETE')
                ORDER BY id DESC LIMIT 1
            """, (ticker,)).fetchone()
        if row:
            return {"id": row[0], "ticker": row[1], "started_date": row[2],
                    "closed_date": row[3], "state": row[4], "notes": row[5]}
        return None

    def get_all_cycles(self, ticker: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, ticker, started_date, closed_date, state, notes
                FROM wheel_cycles WHERE ticker = ? ORDER BY id DESC
            """, (ticker,)).fetchall()
        return [{"id": r[0], "ticker": r[1], "started_date": r[2],
                 "closed_date": r[3], "state": r[4], "notes": r[5]} for r in rows]

    # ------------------------------------------------------------------
    # Leg management
    # ------------------------------------------------------------------

    def add_leg(
        self,
        cycle_id: int,
        leg_type: str,
        strike: float,
        premium_per_share: float,
        contracts: int = 1,
        expiry: Optional[str] = None,
        leg_date: Optional[date] = None,
        notes: str = "",
    ) -> int:
        """Add a leg to a cycle. Returns the new leg_id."""
        d = (leg_date or date.today()).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                INSERT INTO wheel_legs
                  (cycle_id, leg_date, leg_type, strike, premium_per_share, contracts, expiry, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cycle_id, d, leg_type, strike, premium_per_share, contracts, expiry, notes,
                  datetime.now().isoformat()))
            conn.commit()
            return cur.lastrowid

    def get_legs(self, cycle_id: int) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, cycle_id, leg_date, leg_type, strike, premium_per_share, contracts, expiry, notes
                FROM wheel_legs WHERE cycle_id = ? ORDER BY leg_date, id
            """, (cycle_id,)).fetchall()
        cols = ["id", "cycle_id", "leg_date", "leg_type", "strike",
                "premium_per_share", "contracts", "expiry", "notes"]
        return [dict(zip(cols, r)) for r in rows]

    def delete_leg(self, leg_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM wheel_legs WHERE id = ?", (leg_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # High-level: load a full WheelCycle object
    # ------------------------------------------------------------------

    def load_cycle(self, cycle_id: int) -> Optional["WheelCycle"]:  # noqa: F821
        from strategy.wheel import WheelCycle, WheelLeg, WheelState, LegType
        cycle_row = None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, ticker, started_date, closed_date, state, notes FROM wheel_cycles WHERE id = ?",
                (cycle_id,)
            ).fetchone()
            if not row:
                return None
            cycle_row = row

        legs_data = self.get_legs(cycle_id)
        legs = []
        for ld in legs_data:
            try:
                legs.append(WheelLeg(
                    id=ld["id"],
                    cycle_id=ld["cycle_id"],
                    leg_date=date.fromisoformat(ld["leg_date"]),
                    leg_type=LegType(ld["leg_type"]),
                    strike=ld["strike"],
                    premium_per_share=ld["premium_per_share"],
                    contracts=ld["contracts"],
                    expiry=ld["expiry"],
                    notes=ld["notes"] or "",
                ))
            except Exception:
                continue

        return WheelCycle(
            id=cycle_row[0],
            ticker=cycle_row[1],
            started_date=date.fromisoformat(cycle_row[2]),
            closed_date=date.fromisoformat(cycle_row[3]) if cycle_row[3] else None,
            state=WheelState(cycle_row[4]),
            legs=legs,
            notes=cycle_row[5] or "",
        )

    def load_active_cycle(self, ticker: str) -> Optional["WheelCycle"]:  # noqa: F821
        active = self.get_active_cycle(ticker)
        if not active:
            return None
        return self.load_cycle(active["id"])


# ---------------------------------------------------------------------------
# Position ledger  (shares, warrants, long calls)
# ---------------------------------------------------------------------------

class PositionStore:
    """
    Stores your actual holdings — shares, warrants, long calls.
    Lives in gme_intel.db which is gitignored, so positions stay private.

    This is separate from the Wheel tracker (which tracks short options strategy).
    Here you log what you actually own so the dashboard can show real P&L.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker       TEXT NOT NULL DEFAULT 'GME',
                    position_type TEXT NOT NULL,  -- 'shares' | 'warrant' | 'call'
                    quantity     REAL NOT NULL,   -- shares, warrants, or contracts
                    cost_basis   REAL,            -- per share / per warrant / per contract (total paid)
                    strike       REAL,            -- warrants and calls only
                    expiry       TEXT,            -- warrants and calls only (YYYY-MM-DD)
                    notes        TEXT DEFAULT '',
                    created_at   TEXT NOT NULL,
                    updated_at   TEXT NOT NULL
                )
            """)
            conn.commit()

    def upsert_position(
        self,
        position_type: str,   # 'shares' | 'warrant' | 'call'
        quantity: float,
        cost_basis: float = 0.0,
        strike: float = None,
        expiry: str = None,
        notes: str = "",
        ticker: str = "GME",
        position_id: int = None,
    ) -> int:
        """Insert new or update existing position. Returns position id."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            if position_id:
                conn.execute("""
                    UPDATE positions SET quantity=?, cost_basis=?, strike=?, expiry=?,
                    notes=?, updated_at=? WHERE id=?
                """, (quantity, cost_basis, strike, expiry, notes, now, position_id))
                conn.commit()
                return position_id
            else:
                cur = conn.execute("""
                    INSERT INTO positions (ticker, position_type, quantity, cost_basis,
                    strike, expiry, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker, position_type, quantity, cost_basis, strike, expiry, notes, now, now))
                conn.commit()
                return cur.lastrowid

    def get_positions(self, ticker: str = "GME") -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT id, ticker, position_type, quantity, cost_basis, strike, expiry, notes
                FROM positions WHERE ticker = ? ORDER BY position_type, id
            """, (ticker,)).fetchall()
        cols = ["id", "ticker", "position_type", "quantity", "cost_basis", "strike", "expiry", "notes"]
        return [dict(zip(cols, r)) for r in rows]

    def delete_position(self, position_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM positions WHERE id = ?", (position_id,))
            conn.commit()

    def summary(self, current_price: float, chain_snapshot=None) -> dict:
        """
        Compute current portfolio value and P&L across all position types.
        chain_snapshot used to mark-to-market long calls if provided.
        """
        positions = self.get_positions()
        from datetime import date as date_type

        total_cost = 0.0
        total_market_value = 0.0
        shares_qty = 0.0
        shares_avg_cost = 0.0

        breakdown = []

        for p in positions:
            ptype = p["position_type"]
            qty = p["quantity"] or 0
            cb = p["cost_basis"] or 0
            strike = p["strike"]
            expiry = p["expiry"]

            if ptype == "shares":
                cost = qty * cb
                mkt = qty * current_price
                pnl = mkt - cost
                breakdown.append({
                    "Type": "Shares", "Qty": qty,
                    "Avg Cost": f"${cb:.2f}", "Mkt Value": f"${mkt:,.2f}",
                    "P&L": f"${pnl:+,.2f}", "P&L %": f"{pnl/cost*100:+.1f}%" if cost else "—",
                    "Notes": p["notes"] or "", "id": p["id"],
                })
                total_cost += cost
                total_market_value += mkt
                shares_qty += qty
                shares_avg_cost = cb  # last share position's avg cost

            elif ptype == "warrant":
                cost = qty * cb
                # Warrant value = max(spot - strike, 0) intrinsic + time value (approximate)
                intrinsic = max(current_price - (strike or 0), 0)
                mkt = qty * intrinsic  # rough floor — ignores time value
                pnl = mkt - cost
                breakdown.append({
                    "Type": "Warrant", "Qty": qty,
                    "Strike": f"${strike:.2f}" if strike else "—",
                    "Expiry": expiry or "—",
                    "Cost/unit": f"${cb:.2f}",
                    "Intrinsic": f"${intrinsic:.2f}",
                    "Total Cost": f"${cost:,.2f}",
                    "Notes": p["notes"] or "", "id": p["id"],
                })
                total_cost += cost

            elif ptype == "call":
                cost = qty * cb * 100  # cb = premium per share, contract = 100 shares
                # Mark to market from chain if available
                mkt_price = None
                if chain_snapshot and strike and expiry:
                    matching = [
                        c for c in chain_snapshot.contracts
                        if c.option_type == "call"
                        and abs(c.strike - strike) < 0.5
                        and c.expiry == expiry
                    ]
                    if matching:
                        mkt_price = matching[0].mid or matching[0].last

                mkt = (qty * mkt_price * 100) if mkt_price else None
                pnl = (mkt - cost) if mkt is not None else None
                dte = (date_type.fromisoformat(expiry) - date_type.today()).days if expiry else None

                breakdown.append({
                    "Type": "Long Call", "Qty": f"{qty:.0f} contracts",
                    "Strike": f"${strike:.2f}" if strike else "—",
                    "Expiry": expiry or "—",
                    "DTE": dte,
                    "Cost/contract": f"${cb*100:.2f}",
                    "Total Cost": f"${cost:,.2f}",
                    "Mkt Value": f"${mkt:,.2f}" if mkt is not None else "—",
                    "P&L": f"${pnl:+,.2f}" if pnl is not None else "—",
                    "Notes": p["notes"] or "", "id": p["id"],
                })
                total_cost += cost
                if mkt:
                    total_market_value += mkt

        return {
            "breakdown": breakdown,
            "total_cost": total_cost,
            "total_market_value": total_market_value,
            "total_pnl": total_market_value - total_cost,
            "shares_qty": shares_qty,
            "shares_avg_cost": shares_avg_cost,
            "position_count": len(positions),
        }
