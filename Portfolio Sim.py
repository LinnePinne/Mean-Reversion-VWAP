import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================



markets = [
    {
        "name": "EURUSD",
        "csv": "EURUSD_5M_2012-now.csv",
        "pip_size": 0.0001,
        "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.12, "fixed_spread_pips": 0.45, "comm_pips_per_side": 0.25,
                            #0.08 0.12                 #0.30 0.45
        },
        "session_start": "22:00:00",
        "session_end": "03:00:00",
        "vwap_reset": "21:00:00",
    },
    {
        "name": "GBPUSD", "csv": "GBPUSD_5M_2012-2026.csv", "pip_size": 0.0001, "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.14, "fixed_spread_pips": 0.55, "comm_pips_per_side": 0.25, },
                            #0.10 0.14                  #0.40 0.55
        "session_start": "22:00:00",
        "session_end": "03:00:00",
        "vwap_reset": "21:00:00",
    },
    {
        "name": "USDCHF", "csv": "USDCHF_5M_2012-now.csv", "pip_size": 0.0001, "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.16, "fixed_spread_pips": 0.60, "comm_pips_per_side": 0.25, },
                            #0.12 0.16                   #0.45 0.60
        "session_start": "22:00:00",
        "session_end": "03:00:00",
        "vwap_reset": "21:00:00",
    },
    {
        "name": "USDCAD", "csv": "USDCAD_5M_2012-2026.csv", "pip_size": 0.0001, "spread_points_per_pip": 10.0,
        "cost_model": {
            "slippage_pips": 0.16, "fixed_spread_pips": 0.65, "comm_pips_per_side": 0.25, },
                            #0.12 0.16                  #0.50 0.65
        "session_start": "22:00:00",
        "session_end": "03:00:00",
        "vwap_reset": "21:00:00",
    },
]

HALF = 0.5

def clamp_time_series_index_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ unik, sorterad datetime-index."""
    df = df.sort_index()
    if df.index.has_duplicates:
        df = (df.groupby(df.index)
              .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
              .sort_index())
    return df

def annualize_factor_from_resample(rule: str) -> float:
    """Ann-faktor för returns baserat på resample-regel."""
    rule = rule.upper()
    if rule in ("D", "1D"):
        return 365.0
    if rule in ("B", "1B"):
        return 252.0
    # fallback: tolka som dagar
    return 365.0

def compute_stats_from_trades(trades_df: pd.DataFrame) -> dict:
    """
    Samma logik som era totala stats, men på en subset av trades.
    trades_df måste ha kolumner: pnl, equity (valfritt), is_win (valfritt)
    """
    if trades_df.empty:
        return {}

    df_ = trades_df.copy()

    # equity behövs för DD
    df_["equity"] = df_["pnl"].cumsum()

    df_["is_win"] = df_["pnl"] > 0

    gross_profit = df_.loc[df_["pnl"] > 0, "pnl"].sum()
    gross_loss = df_.loc[df_["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = df_.loc[df_["pnl"] > 0, "pnl"].mean()
    avg_loss = df_.loc[df_["pnl"] < 0, "pnl"].mean()

    winrate = df_["is_win"].mean()
    expectancy = df_["pnl"].mean()

    roll_max = df_["equity"].cummax()
    dd = df_["equity"] - roll_max
    max_dd_points = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Losing streak
    loss_streak = 0
    max_loss_streak = 0
    for is_win in df_["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    pnl_std = df_["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(df_)) if pnl_std and pnl_std > 0 else np.nan

    return {
        "Trades": int(len(df_)),
        "Total PnL (points)": float(df_["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }


def run_backtest_for_market(
    market_name: str,
    csv_path: str,
    pip_size: float,
    spread_points_per_pip: float = 10.0,
    cost_model: dict | None = None,
    session_start: str = "22:00:00",
    session_end: str = "03:00:00",
    vwap_reset: str = "21:00:00",
):
    global entry_mid
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name} ")
    print("=" * 70 + "\n")

    # 1) Ladda data
    df = pd.read_csv(csv_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()
    # ---- DEDUPE INDEX (viktigt för groupby/transform) ----
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    def pips_to_price(pips: float) -> float:
        return float(pips) * float(pip_size)

    if cost_model is None:
        cost_model = {}

    slippage_pips = float(cost_model.get("slippage_pips"))
    fixed_spread_pips = float(cost_model.get("fixed_spread_pips"))
    comm_pips_per_side = float(cost_model.get("comm_pips_per_side"))

    def commission_round_turn_price() -> float:
        return 2.0 * pips_to_price(comm_pips_per_side)

    def is_usd_quote(symbol: str) -> bool:
        # XXXUSD
        return symbol.endswith("USD") and not symbol.startswith("USD")

    def is_usd_base(symbol: str) -> bool:
        # USDXXX
        return symbol.startswith("USD") and not symbol.endswith("USD")

    def pip_value_usd_per_unit(symbol: str, price: float) -> float:
        """
        USD value of 1 pip for 1 unit of base currency.
        - For XXXUSD: 1 pip = pip_size USD per unit
        - For USDXXX (USDJPY, USDCHF, USDCAD): 1 pip in quote, convert to USD ~ pip_size / price
        """
        if is_usd_quote(symbol):
            return float(pip_size)
        # USD base or cross where quote != USD: approximate conversion using price
        return float(pip_size) / float(price)

    # volymkolumn
    if "volume" in df.columns:
        vol_col = "volume"
    elif "tick_volume" in df.columns:
        vol_col = "tick_volume"
    else:
        df["volume_dummy"] = 1.0
        vol_col = "volume_dummy"

    def compute_session_anchored_vwap_and_std(data: pd.DataFrame, vol_col: str, reset_time: str = "21:00:00"):
        df_v = data.copy()

        tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3.0
        vol = df_v[vol_col].astype(float).fillna(0.0)

        df_v["tp"] = tp
        df_v["vol"] = vol
        df_v["tp_vol"] = df_v["tp"] * df_v["vol"]

        rt = pd.to_datetime(reset_time).time()

        session_date = df_v.index.floor("D")
        session_date = session_date.where(df_v.index.time >= rt, session_date - pd.Timedelta(days=1))
        df_v["session_date"] = session_date

        g = df_v.groupby("session_date", sort=False)

        df_v["cum_tp_vol"] = g["tp_vol"].cumsum()
        df_v["cum_vol"] = g["vol"].cumsum()

        vwap = df_v["cum_tp_vol"] / df_v["cum_vol"].replace(0.0, np.nan)

        # transform är OK när index är unikt (därför dedupe före denna funktion)
        std = g["tp"].transform(lambda x: x.expanding().std(ddof=0))

        return vwap, std

    df["VWAP"], df["TP_STD"] = compute_session_anchored_vwap_and_std(df, vol_col, vwap_reset)
    df["VWAP_prev"] = df["VWAP"].shift(1)
    df["TP_STD_prev"] = df["TP_STD"].shift(1)

    session_start_t = pd.to_datetime(session_start).time()
    session_end_t = pd.to_datetime(session_end).time()

    def in_session(ts) -> bool:
        t = ts.time()

        # Normal session (ex 01:00 -> 07:00)
        if session_start_t < session_end_t:
            return (t >= session_start_t) and (t < session_end_t)

        # Wrap session (ex 23:00 -> 03:00)
        # Då är det "i session" om tiden är >= start ELLER < end
        return (t >= session_start_t) or (t < session_end_t)

    USE_SPREAD_PIPS_COL = 'spread_pips' in df.columns
    USE_SPREAD_POINTS_COL = 'spread_points' in df.columns

    def get_spread_pips(row) -> float:
        if USE_SPREAD_PIPS_COL:
            return float(row['spread_pips'])
        if USE_SPREAD_POINTS_COL:
            return float(row['spread_points']) / float(spread_points_per_pip)
        return float(fixed_spread_pips)


    std_mult = 2.6  # hur många std från VWAP krävs

    # 4) Backtest-loop
    trades = []
    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):

        ts = idx_list[i]
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # ======================
        # EXIT-logik (signal på bar i, fill på bar i+1 open)
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            if pos_direction == 'LONG':
                if row["close"] >= row["VWAP"]:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)

                    exit_price = next_row["open"] - HALF * spread_px - slip_px
                    exit_reason = 'vwap_exit'

            else:  # SHORT
                if row["close"] <= row["VWAP"]:
                    spread_pips = get_spread_pips(next_row)
                    spread_px = pips_to_price(spread_pips)
                    slip_px = pips_to_price(slippage_pips)

                    exit_price = next_row["open"] + HALF * spread_px + slip_px

            if exit_price is not None:
                exit_time = idx_list[i + 1]  # matchar fill på next_row open
                comm_px = commission_round_turn_price()

                if pos_direction == 'LONG':
                    pnl = (exit_price - entry_price) - comm_px
                else:
                    pnl = (entry_price - exit_price) - comm_px

                exit_mid = float(next_row["open"])

                trades.append({
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "Direction": pos_direction,
                    "Entry Price": entry_price,  # executed (inkl costs)
                    "Exit Price": exit_price,  # executed (inkl costs)
                    "Entry Mid": entry_mid,  # mid approx
                    "Exit Mid": exit_mid,  # mid approx
                    "pnl": pnl,  # executed pnl per 1 unit i price-termer
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None
                # entry_index = None

                continue  # viktigt: undvik att gå in i ny trade samma iteration

        # Sessionfilter: både signalbar och fillbar måste vara i session
        if not in_session(ts) or not in_session(idx_list[i + 1]):
            continue

        # hoppa entry-logik om vi fortfarande är i trade
        if in_position:
            continue
        # ======================
        # ENTRY-logik
        # ======================
        close_price = row["close"]
        prev_close = prev_row["close"]
        next_open = next_row['open']
        vwap_prev = row["VWAP_prev"]
        std_prev = row["TP_STD_prev"]
        vwap = row["VWAP"]
        std = row["TP_STD"]

        if not np.isfinite(vwap_prev) or not np.isfinite(std_prev) or std_prev == 0:
            continue

        # STD-bands kring VWAP
        upper_band = vwap + std_mult * std
        lower_band = vwap - std_mult * std
        upper_band_prev = vwap_prev + std_mult * std_prev
        lower_band_prev = vwap_prev - std_mult * std_prev

        upper_band_break = (prev_close < upper_band_prev) and (close_price > upper_band)
        lower_band_break = (prev_close > lower_band_prev) and (close_price < lower_band)


        long_signal = lower_band_break
        short_signal = upper_band_break

        if long_signal:
            pos_direction = 'LONG'
            entry_time = idx_list[i + 1]  # fill sker på nästa bar open

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_mid = float(next_row["open"])
            entry_price = entry_mid + HALF * spread_px + slip_px
            in_position = True

        elif short_signal:
            pos_direction = 'SHORT'
            entry_time = idx_list[i + 1]

            spread_pips = get_spread_pips(next_row)
            spread_px = pips_to_price(spread_pips)
            slip_px = pips_to_price(slippage_pips)

            entry_mid = float(next_row["open"])
            entry_price = entry_mid - HALF * spread_px - slip_px
            in_position = True

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # PLOT (som du redan får)
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return stats, trades_df

def simulate_portfolio_equal_risk(
    trades: pd.DataFrame,
    market_cfg: dict,
    start_capital: float = 50_000.0,
    max_one_position_per_market: bool = True,
    exposure_scale: float = 6.0,
    weights: dict | None = None,
):
    """
    Equal risk per market (Variant B): varje marknad får lika stor notional-budget vid entry.
    - Vi antar att trades har Entry/Exit Price som redan inkluderar spread+slippage.
    - 'pnl' i trades är price-delta per 1 unit (netto efter comm_px).
    """

    df = trades.copy()
    df = df.sort_values("Entry Time").reset_index(drop=True)

    markets_list = sorted(df["Market"].unique().tolist())
    n_markets = len(markets_list)
    if n_markets == 0:
        return None, None, None

    if weights is None:
        market_budget = {m: 1.0 / n_markets for m in markets_list}
    else:
        market_budget = {m: float(weights.get(m, 0.0)) for m in markets_list}
        s = sum(market_budget.values())
        if s <= 0:
            market_budget = {m: 1.0 / n_markets for m in markets_list}
        else:
            market_budget = {m: market_budget[m] / s for m in markets_list}

    equity = start_capital
    equity_curve = []
    open_pos = {m: None for m in markets_list}  # store dict with units, entry, etc.

    # Event queue: vi itererar på entries men stänger allt som exit:ar före nästa entry
    # Skapa exits per market som min-heap-ish via scanning
    # Enkelt: vi processar i tidsordning med en lista över alla exit-events.
    exit_events = df[["Exit Time", "Market"]].copy()
    exit_events["idx"] = exit_events.index
    exit_events = exit_events.sort_values("Exit Time").reset_index(drop=True)
    exit_ptr = 0


    def usd_pnl_for_trade(row, units: float) -> float:
        sym = row["Market"]
        pip_size = float(market_cfg[sym]["pip_size"])

        # robust pip-value conversion based on exit price (good approximation)
        exit_price = float(row["Exit Price"])

        # determine USD pip value per unit
        if sym.endswith("USD") and not sym.startswith("USD"):
            pip_value = pip_size
        else:
            pip_value = pip_size / exit_price

        # Convert price-delta per unit -> pips -> USD
        # pnl_price_per_unit already includes commission in price units.
        pnl_price = float(row["pnl"])
        pips_move = pnl_price / pip_size
        return pips_move * pip_value * units

    trade_log = []

    for i, row in df.iterrows():
        t_entry = row["Entry Time"]

        # 1) Close any positions whose exit time <= this entry time
        while exit_ptr < len(exit_events) and exit_events.loc[exit_ptr, "Exit Time"] <= t_entry:
            idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
            r_close = df.loc[idx_to_close]
            mkt = r_close["Market"]

            pos = open_pos.get(mkt)

            if pos is not None and pos.get("trade_idx") == idx_to_close:
                units = pos["units"]
                pnl_usd = usd_pnl_for_trade(r_close, units)
                equity += pnl_usd

                trade_log.append({
                    "Market": mkt,
                    "Entry Time": r_close["Entry Time"],
                    "Exit Time": r_close["Exit Time"],
                    "Direction": r_close.get("Direction", None),
                    "Entry Price": r_close.get("Entry Price", np.nan),
                    "Exit Price": r_close.get("Exit Price", np.nan),
                    "Entry Mid": r_close.get("Entry Mid", np.nan),
                    "Units": units,
                    "PnL_USD": pnl_usd,
                    "Equity": equity,
                })

                open_pos[mkt] = None
                equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})
            else:
                # Exit-event för en trade som aldrig öppnades i portföljen (skippad entry)
                # eller mismatch -> ignorera
                pass

            exit_ptr += 1

        mkt = row["Market"]
        if max_one_position_per_market and open_pos.get(mkt) is not None:
            # redan i position, skip entry
            continue

        # 2) Size: equal budget per market at time of entry
        notional_usd = equity * market_budget[mkt] * exposure_scale

        entry_price = float(row["Entry Price"])

        # Units sizing:
        # - For XXXUSD: units = notional / price
        # - For USDXXX: base is USD so units ~ notional
        if mkt.startswith("USD") and not mkt.endswith("USD"):
            units = notional_usd
        else:
            units = notional_usd / entry_price

        open_pos[mkt] = {"units": units, "trade_idx": i}

        equity_curve.append({"Time": t_entry, "Equity": equity})

    # 3) Close remaining positions at their exits
    # process remaining exits in chronological order
    while exit_ptr < len(exit_events):
        idx_to_close = int(exit_events.loc[exit_ptr, "idx"])
        r_close = df.loc[idx_to_close]
        mkt = r_close["Market"]

        pos = open_pos.get(mkt)

        # Stäng bara om den trade som exit:ar är exakt den du öppnade
        if pos is not None and pos.get("trade_idx") == idx_to_close:
            units = pos["units"]
            pnl_usd = usd_pnl_for_trade(r_close, units)
            equity += pnl_usd

            trade_log.append({
                "Market": mkt,
                "Entry Time": r_close["Entry Time"],
                "Exit Time": r_close["Exit Time"],
                "Direction": r_close.get("Direction", None),
                "Entry Price": r_close.get("Entry Price", np.nan),
                "Exit Price": r_close.get("Exit Price", np.nan),
                "Units": units,
                "PnL_USD": pnl_usd,
                "Equity": equity,
                "Entry Mid": r_close.get("Entry Mid", np.nan),
                "Direction": r_close.get("Direction", None),
            })

            open_pos[mkt] = None

            # NYTT: logga equity vid exit (där equity faktiskt ändras)
            equity_curve.append({"Time": r_close["Exit Time"], "Equity": equity})
        exit_ptr += 1

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset=["Time"]).sort_values("Time")
    log_df = pd.DataFrame(trade_log).sort_values("Exit Time")

    # Drawdown stats
    if not eq_df.empty:
        eq_df["RollMax"] = eq_df["Equity"].cummax()
        eq_df["DD_$"] = eq_df["Equity"] - eq_df["RollMax"]
        eq_df["DD_%"] = eq_df["DD_$"] / eq_df["RollMax"]
        max_dd_usd = float(eq_df["DD_$"].min())
        max_dd_pct = float(eq_df["DD_%"].min())
    else:
        max_dd_usd = 0.0
        max_dd_pct = 0.0

    # ==========================
    # PORTFÖLJ-RATIO: Sharpe / Sortino / Calmar
    # ==========================
    days = 0.0
    years = 0.0

    sharpe = np.nan
    sortino = np.nan
    calmar = np.nan
    cagr = np.nan

    if not eq_df.empty and len(eq_df) >= 2:
        eq_ts = eq_df.copy()
        eq_ts["Time"] = pd.to_datetime(eq_ts["Time"])
        eq_ts = eq_ts.sort_values("Time").set_index("Time")

        # Daglig equity (kalenderdagar) och forward fill
        daily_eq = eq_ts["Equity"].resample("D").last().ffill()
        # Dagliga returns
        daily_ret = daily_eq.pct_change().dropna()

        #print("Days total:", len(daily_ret))
        #print("Days non-zero:", (daily_ret != 0).sum())
        #print("Share non-zero:", (daily_ret != 0).mean())

        if len(daily_ret) >= 30:
            ann_factor = 365.0  # eftersom vi använder kalenderdagar ("D")

            ret_mean = daily_ret.mean()
            ret_std = daily_ret.std(ddof=1)

            # Sharpe (RF=0)
            sharpe = (ret_mean / ret_std) * np.sqrt(ann_factor) if ret_std and ret_std > 0 else np.nan

            # Sortino (MAR = 0): downside deviation = sqrt(mean(min(0, r)^2))
            mar = 0.0
            downside_sq = np.minimum(0.0, daily_ret - mar) ** 2
            downside_dev = np.sqrt(downside_sq.mean())

            sortino = (ret_mean / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

            # CAGR (använd exakt tidslängd)
            start_val = float(daily_eq.iloc[0])
            end_val = float(daily_eq.iloc[-1])
            days = (daily_eq.index[-1] - daily_eq.index[0]).total_seconds() / 86400.0

            if days > 0 and start_val > 0:
                cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
                years = days / 365.0 if days > 0 else 0.0

            # Calmar = CAGR / MaxDD (fraction)
            max_dd_frac = abs(max_dd_pct)  # max_dd_pct är en fraction (t.ex. -0.013)
            calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan

    summary = {
        "Start Capital": start_capital,
        "End Equity": float(equity),
        "Net PnL ($)": float(equity - start_capital),
        "Return (%)": float((equity / start_capital - 1.0) * 100.0),
        "Max Drawdown ($)": abs(max_dd_usd),
        "Max Drawdown (%)": abs(max_dd_pct) * 100.0,
        "Markets": n_markets,
        "Trades Closed": int(len(log_df)),
        "Days": float(days),
        "Years": float(years),
        "Sharpe (daily, ann.)": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino (daily, ann.)": float(sortino) if np.isfinite(sortino) else np.nan,
        "CAGR (%)": float(cagr * 100.0) if np.isfinite(cagr) else np.nan,
        "Calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }

    return summary, eq_df, log_df

def build_price_series(market_cfg: dict, start_time=None, end_time=None, price_col="close"):
    """
    Returnerar dict: symbol -> pd.Series (price_col) med datetimeindex.
    Antag att CSV har 'timestamp' eller 'datetime' och 'close'.
    """
    out = {}
    for sym, cfg in market_cfg.items():
        df = pd.read_csv(cfg["csv"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            raise ValueError(f"{sym}: saknar timestamp/datetime")

        df = df.sort_index()
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")].sort_index()

        s = df[price_col].astype(float)

        if start_time is not None:
            s = s[s.index >= start_time]
        if end_time is not None:
            s = s[s.index <= end_time]

        out[sym] = s
    return out


def usd_pnl_from_price_delta(sym: str, pip_size: float, price_now: float, pnl_price_per_unit: float, units: float) -> float:
    """
    Konverterar price-delta per unit till USD:
    - XXXUSD: 1 pip = pip_size USD per 1 unit => pip_value_per_unit = pip_size
    - USDXXX: pip_value_per_unit ~ pip_size / price_now
    """
    if sym.endswith("USD") and not sym.startswith("USD"):
        pip_value = pip_size
    else:
        pip_value = pip_size / price_now

    pips_move = pnl_price_per_unit / pip_size
    return pips_move * pip_value * units


def approx_usd_per_pip(sym: str, pip_size: float, units: float, price: float) -> float:
    """
    Approx USD value of 1 pip for a given position size (units).
    - XXXUSD: 1 pip = pip_size USD per unit
    - USDXXX: 1 pip in quote -> convert to USD using price => pip_size / price USD per unit
    """
    if sym.endswith("USD") and not sym.startswith("USD"):
        # EURUSD, GBPUSD, AUDUSD, NZDUSD...
        return units * pip_size
    else:
        # USDJPY, USDCHF, USDCAD...
        return units * (pip_size / price)

def compute_portfolio_mtm_equity_and_intraday_dd(
    port_log: pd.DataFrame,
    market_cfg: dict,
    start_capital: float = 50_000.0,
    freq: str = "5min",          # ni har 5m data
    price_col: str = "close",    # MTM på close (kan byta till open)
):
    """
    Bygger MTM equitykurva inklusive orealiserad PnL och räknar intraday DD.
    port_log måste ha: Market, Entry Time, Exit Time, Units, Direction, Entry Price, Entry Mid (minst)
    """

    if port_log.empty:
        return None, None, None

    log = port_log.copy()
    log["Entry Time"] = pd.to_datetime(log["Entry Time"])
    log["Exit Time"] = pd.to_datetime(log["Exit Time"])

    t0 = log["Entry Time"].min()
    t1 = log["Exit Time"].max()

    # 1) Ladda prisserier och resampla till gemensam frekvens
    prices = build_price_series(market_cfg, start_time=t0, end_time=t1, price_col=price_col)

    # 2) Skapa master timeline (union) och forward-fill priser
    # --- NYTT: validera tider ---
    if pd.isna(t0) or pd.isna(t1):
        raise ValueError("MTM: t0/t1 är NaT. Kontrollera att port_log har giltiga Entry/Exit Time.")

    start = pd.Timestamp(t0).floor(freq)
    end = pd.Timestamp(t1).ceil(freq)

    if end < start:
        raise ValueError(f"MTM: end < start ({end} < {start}). Kontrollera tiderna i port_log.")

    master_index = pd.date_range(start=start, end=end, freq=freq)

    if len(master_index) == 0:
        raise ValueError("MTM: master_index blev tom. Kontrollera freq och tidsintervall.")

    price_df = pd.DataFrame(index=master_index)
    for sym, s in prices.items():
        # align/resample till master
        aligned = s.reindex(master_index).ffill()
        price_df[sym] = aligned

    # 3) Event lists för att öppna/stänga positioner
    #    Vi itererar tid och håller en dict med öppna positioner per market (eller flera om ni tillåter)
    opens = log.sort_values("Entry Time").reset_index(drop=True)
    closes = log.sort_values("Exit Time").reset_index(drop=True)

    o_ptr = 0
    c_ptr = 0

    open_pos = {}  # sym -> list of positions (om ni vill tillåta flera)
    cash_equity = start_capital  # realiserad equity

    mtm_records = []

    for t in master_index:
        # close events
        while c_ptr < len(closes) and closes.loc[c_ptr, "Exit Time"] <= t:
            r = closes.loc[c_ptr]
            sym = r["Market"]

            # realiserad PnL finns redan i er simulator som PnL_USD per trade i port_log?
            # I er port_log har ni PnL_USD och Equity efter stängning.
            # Om den finns: använd den för cash_equity så cash matchar simulatorn.
            if "PnL_USD" in r:
                cash_equity += float(r["PnL_USD"])
            else:
                # fallback: räkna från executed pnl price per unit (om ni sparar den)
                raise ValueError("port_log saknar PnL_USD. Lägg till det i simulate_portfolio_equal_risk.")

            # ta bort positionen ur open_pos
            if sym in open_pos and len(open_pos[sym]) > 0:
                open_pos[sym].pop(0)
                if len(open_pos[sym]) == 0:
                    del open_pos[sym]

            c_ptr += 1

        # open events
        while o_ptr < len(opens) and opens.loc[o_ptr, "Entry Time"] <= t:
            r = opens.loc[o_ptr]
            sym = r["Market"]
            cm = market_cfg[sym].get("cost_model", {})
            pos = {
                "Direction": r.get("Direction", None),
                "Units": float(r["Units"]),
                "EntryPrice": float(r["Entry Price"]) if "Entry Price" in r else np.nan,  # executed
                "EntryMid": float(r["Entry Mid"]) if "Entry Mid" in r else np.nan,  # valfritt, kan behållas
                "SpreadPips": float(cm.get("fixed_spread_pips", 0.0)),
                "SlipPips": float(cm.get("slippage_pips", 0.0)),
            }
            open_pos.setdefault(sym, []).append(pos)
            o_ptr += 1

        # 4) MTM orealiserad PnL
        unreal = 0.0
        for sym, plist in open_pos.items():
            mid_now = float(price_df.loc[t, sym])  # vi behandlar close som mid approx i MTM
            pip_size = float(market_cfg[sym]["pip_size"])

            for pos in plist:
                units = float(pos["Units"])
                direction = pos.get("Direction", None)

                entry_exec = float(pos.get("EntryPrice", np.nan))  # executed entry (inkl costs)
                if not np.isfinite(entry_exec) or not np.isfinite(mid_now):
                    continue

                # Kostnader för att "likvidera" just nu (konservativt)
                spread_pips = float(pos.get("SpreadPips", 0.0))
                slip_pips = float(pos.get("SlipPips", 0.0))

                spread_px = spread_pips * pip_size
                slip_px = slip_pips * pip_size

                if direction == "LONG":
                    # Om vi stänger long nu: vi säljer på bid ≈ mid - halfspread, plus slippage
                    liquidation = mid_now - 0.5 * spread_px - slip_px
                    pnl_price_per_unit = liquidation - entry_exec
                else:
                    # Om vi stänger short nu: vi köper på ask ≈ mid + halfspread, plus slippage
                    liquidation = mid_now + 0.5 * spread_px + slip_px
                    pnl_price_per_unit = entry_exec - liquidation

                unreal += usd_pnl_from_price_delta(sym, pip_size, mid_now, pnl_price_per_unit, units)

        mtm_equity = cash_equity + unreal
        mtm_records.append({
            "Time": t,
            "Equity_MTM": mtm_equity,
            "Cash": cash_equity,
            "Unreal": unreal
        })

    mtm_df = pd.DataFrame(mtm_records).set_index("Time")

    # 5) Intraday drawdown:
    #    (a) Total max DD på 5m
    mtm_df["RollMax"] = mtm_df["Equity_MTM"].cummax()
    mtm_df["DD_$"] = mtm_df["Equity_MTM"] - mtm_df["RollMax"]
    mtm_df["DD_%"] = mtm_df["DD_$"] / mtm_df["RollMax"]

    max_dd_usd = float(mtm_df["DD_$"].min())
    max_dd_pct = float(mtm_df["DD_%"].min())

    #    (b) Intraday max DD: reset peak varje dag
    g = mtm_df.groupby(mtm_df.index.date)
    daily_peak = g["Equity_MTM"].cummax()
    intraday_dd = mtm_df["Equity_MTM"] - daily_peak
    mtm_df["Intraday_DD_$"] = intraday_dd
    mtm_df["Intraday_DD_%"] = intraday_dd / daily_peak

    max_intraday_dd_usd = float(mtm_df["Intraday_DD_$"].min())
    max_intraday_dd_pct = float(mtm_df["Intraday_DD_%"].min())

    dd_summary = {
        "Max DD MTM ($)": abs(max_dd_usd),
        "Max DD MTM (%)": abs(max_dd_pct) * 100.0,
        "Max Intraday DD MTM ($)": abs(max_intraday_dd_usd),
        "Max Intraday DD MTM (%)": abs(max_intraday_dd_pct) * 100.0,
    }
    print(sym, cm)
    return dd_summary, mtm_df

def compute_risk_metrics_from_equity(eq_series: pd.Series, resample_rule: str = "D") -> dict:
    """
    eq_series: pd.Series med datetimeindex och equity-värden.
    Returnerar sharpe/sortino/cagr/calmar + maxdd.
    """
    out = {
        "Sharpe (daily, ann.)": np.nan,
        "Sortino (daily, ann.)": np.nan,
        "CAGR (%)": np.nan,
        "Calmar": np.nan,
        "Max Drawdown ($)": np.nan,
        "Max Drawdown (%)": np.nan,
        "Days": np.nan,
        "Years": np.nan,
    }
    if eq_series is None or eq_series.empty or len(eq_series) < 2:
        return out

    eq = eq_series.copy()
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()

    # Resample till daglig equity (eller valfri regel), forward fill
    eq_r = eq.resample(resample_rule).last().ffill()
    ret = eq_r.pct_change().dropna()
    if len(ret) < 30:
        # för få datapunkter för stabila annualiserade mått
        # men vi kan ändå räkna DD och CAGR
        pass

    # Drawdown
    roll_max = eq_r.cummax()
    dd = eq_r - roll_max
    max_dd_usd = float(dd.min())
    max_dd_pct = float((dd / roll_max).min())

    out["Max Drawdown ($)"] = abs(max_dd_usd)
    out["Max Drawdown (%)"] = abs(max_dd_pct) * 100.0

    ann_factor = annualize_factor_from_resample(resample_rule)

    # Sharpe / Sortino
    if len(ret) >= 30:
        mu = ret.mean()
        sd = ret.std(ddof=1)
        sharpe = (mu / sd) * np.sqrt(ann_factor) if sd and sd > 0 else np.nan

        mar = 0.0
        downside_sq = np.minimum(0.0, ret - mar) ** 2
        downside_dev = np.sqrt(downside_sq.mean())
        sortino = (mu / downside_dev) * np.sqrt(ann_factor) if downside_dev and downside_dev > 0 else np.nan

        out["Sharpe (daily, ann.)"] = float(sharpe) if np.isfinite(sharpe) else np.nan
        out["Sortino (daily, ann.)"] = float(sortino) if np.isfinite(sortino) else np.nan


    # CAGR
    start_val = float(eq_r.iloc[0])
    end_val = float(eq_r.iloc[-1])
    days = (eq_r.index[-1] - eq_r.index[0]).total_seconds() / 86400.0
    years = days / 365.0 if days > 0 else np.nan
    out["Days"] = float(days) if np.isfinite(days) else np.nan
    out["Years"] = float(years) if np.isfinite(years) else np.nan

    if days > 0 and start_val > 0:
        cagr = (end_val / start_val) ** (ann_factor / days) - 1.0
        out["CAGR (%)"] = float(cagr * 100.0) if np.isfinite(cagr) else np.nan

        max_dd_frac = abs(max_dd_pct)
        calmar = (cagr / max_dd_frac) if (np.isfinite(cagr) and max_dd_frac and max_dd_frac > 0) else np.nan
        out["Calmar"] = float(calmar) if np.isfinite(calmar) else np.nan

    return out

def build_daily_returns_matrix_from_port_log(
    port_log: pd.DataFrame,
    markets: list[str],
    start_capital: float = 50_000.0,
    date_col: str = "Exit Time",
    pnl_col: str = "PnL_USD",
) -> pd.DataFrame:
    """
    Skapar daily returns per market från port_log (realiserad PnL vid Exit Time).
    Returnerar df: index=Date, columns=markets.
    """
    log = port_log.copy()
    log[date_col] = pd.to_datetime(log[date_col])
    log["Date"] = log[date_col].dt.floor("D")

    daily_pnl = (
        log.groupby(["Date", "Market"])[pnl_col]
        .sum()
        .unstack("Market")
        .reindex(columns=markets)
        .fillna(0.0)
    )

    # Enkel normalisering (Carver-style baseline): dela med startkapital
    daily_ret = daily_pnl / float(start_capital)
    return daily_ret


def erc_weights_from_cov_long_only(cov: pd.DataFrame, iters: int = 2000) -> dict:
    assets = cov.columns.tolist()
    n = len(assets)
    C = cov.values

    # start equal
    w = np.ones(n) / n

    for _ in range(iters):
        # portfolio risk
        sigma_p = np.sqrt(w @ C @ w)
        if sigma_p <= 0:
            break

        # marginal contribution to risk
        mrc = (C @ w) / sigma_p

        # risk contributions
        rc = w * mrc
        target = rc.mean()

        # multiplicative update
        w = w * (target / (rc + 1e-12))

        # normalize
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()

    if _ % 500 == 0:
        print("iter", _, "w:", dict(zip(assets, w)))

    return {assets[i]: float(w[i]) for i in range(n)}


# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []

def iter_market_items(markets_obj):
    # Tillåt både list-of-dicts och dict(name->cfg)
    if isinstance(markets_obj, dict):
        for name, cfg in markets_obj.items():
            # om cfg redan är dict med csv/pip_size osv
            if isinstance(cfg, dict):
                cfg = cfg.copy()
                cfg.setdefault("name", name)
                yield cfg
            else:
                # om någon råkat lägga en str här
                yield {"name": str(name), "csv": None, "pip_size": None}
    else:
        for item in markets_obj:
            yield item

for m in iter_market_items(markets):
    try:
        if not isinstance(m, dict):
            raise TypeError(f"Market config är inte dict: {type(m)} value={m}")

        stats, trades_df = run_backtest_for_market(
            m["name"],
            m["csv"],
            m["pip_size"],
            m.get("spread_points_per_pip", 10.0),
            cost_model=m.get("cost_model", None),
            session_start=m.get("session_start", "22:00:00"),
            session_end=m.get("session_end", "03:00:00"),
            vwap_reset=m.get("vwap_reset", "21:00:00"),
        )

        if stats is not None and trades_df is not None:
            trades_df["Market"] = m["name"]
            all_results.append(stats)
            all_trades.append(trades_df)

    except Exception as e:
        name = m.get("name", str(m)) if isinstance(m, dict) else str(m)
        csv_ = m.get("csv", "?") if isinstance(m, dict) else "?"
        print(f"\n*** FEL för {name} ({csv_}): {e}\n")

# ==========================
# PORTFÖLJ: samla trades + market lookup
# ==========================

if all_trades:
    portfolio_trades = pd.concat(all_trades, ignore_index=True)
    portfolio_trades["Entry Time"] = pd.to_datetime(portfolio_trades["Entry Time"])
    portfolio_trades["Exit Time"]  = pd.to_datetime(portfolio_trades["Exit Time"])

    market_cfg = {m["name"]: m for m in markets}

    port_summary, port_eq, port_log = simulate_portfolio_equal_risk(
        portfolio_trades,
        market_cfg,
        start_capital=50_000.0,
        max_one_position_per_market=True,
    )

    '''
    # ==========================
    # SANITY CHECKS (måste matcha)
    # ==========================


    plt.figure(figsize=(12, 5))
    plt.plot(port_eq["Time"], port_eq["Equity"])
    plt.title("Portfolio Equity Curve ($)")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

    dd_summary, mtm_df = compute_portfolio_mtm_equity_and_intraday_dd(
        port_log,
        market_cfg,
        start_capital=50_000.0,
        freq="5min",
        price_col="close",
    )

    # Lägg in i port_summary
    port_summary.update(dd_summary)

    print("\n--- MTM / INTRADAY DD ---")
    for k, v in dd_summary.items():
        print(f"{k}: {v:.4f}")

    # ===== NYTT: Riskmått från MTM equity istället för "stegig" realiserad equity =====
    mtm_metrics = compute_risk_metrics_from_equity(mtm_df["Equity_MTM"], resample_rule="D")

    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    print("Sum PnL_USD:", float(port_log["PnL_USD"].sum()))
    print("Equity change:", float(port_summary["End Equity"] - port_summary["Start Capital"]))
    print("Diff:", float(port_log["PnL_USD"].sum() - (port_summary["End Equity"] - port_summary["Start Capital"])))

    print("Avg Units:", port_log["Units"].mean())
    print("Median Units:", port_log["Units"].median())

    pip_size = market_cfg["EURUSD"]["pip_size"]
    pips_per_trade = (portfolio_trades["pnl"] / pip_size)
    print("Avg pips/trade:", pips_per_trade.mean())
    print("Median pips/trade:", pips_per_trade.median())

    print("\n$ per pip (approx) per market @ avg units (using median Entry Price):")
    for sym in sorted(port_log["Market"].unique()):
        sub = port_log[port_log["Market"] == sym].dropna(subset=["Entry Price"])
        if sub.empty:
            continue

        avg_units_sym = float(sub["Units"].mean())
        pip_size_sym = float(market_cfg[sym]["pip_size"])
        price_sym = float(sub["Entry Price"].median())

        usd_per_pip_sym = approx_usd_per_pip(sym, pip_size_sym, avg_units_sym, price_sym)
        print(f"{sym}: {usd_per_pip_sym:.4f}")

    # --- Correct $/pip sanity check (works for all majors) ---
    if not port_log.empty:
        avg_units = float(port_log["Units"].mean())

        # välj en representativ trade för att få pris (median entry price)
        sample = port_log.dropna(subset=["Entry Price"]).copy()
        if not sample.empty:
            sym0 = sample["Market"].iloc[0]
            pip_size0 = float(market_cfg[sym0]["pip_size"])
            price0 = float(sample["Entry Price"].median())

            usd_per_pip = approx_usd_per_pip(sym0, pip_size0, avg_units, price0)
            print("Approx $ per pip @ avg units:", usd_per_pip)
        else:
            print("Approx $ per pip @ avg units: N/A (no Entry Price)")

    print("\n" + "="*70)
    print(" PORTFÖLJ-RESULTAT (USD) ")
    print("="*70)

    # Skriv över portföljens riskmått så de blir realistiska
    for k in ["Sharpe (daily, ann.)", "Sortino (daily, ann.)", "CAGR (%)", "Calmar", "Days", "Years",
              "Max Drawdown ($)", "Max Drawdown (%)"]:
        if k in mtm_metrics:
            port_summary[k] = mtm_metrics[k]

    for k, v in port_summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ==========================
    # ERC WEIGHTS (Carver-style)
    # ==========================

    erc_markets = sorted(port_log["Market"].unique().tolist())

    # 1) daily returns matrix per market från equal-weight port_log
    daily_ret = build_daily_returns_matrix_from_port_log(
        port_log=port_log,
        markets=erc_markets,
        start_capital=50_000.0,
    )

    daily_ret = daily_ret.sort_index()
    daily_ret = daily_ret.asfreq("D", fill_value=0.0)  # fyll 0 på dagar utan exits

    # 2) covariance
    cov = daily_ret.cov()

    # 3) ERC weights (long-only)
    erc_w = erc_weights_from_cov_long_only(cov, iters=2000)

    w = np.array([erc_w[m] for m in erc_markets])
    C = cov.values
    sigma_p = np.sqrt(w @ C @ w)
    mrc = (C @ w) / sigma_p
    rc = w * mrc


    print("\nRisk contributions:")
    for i, m in enumerate(erc_markets):
        print(m, rc[i])
    print("RC ratio max/min:", rc.max() / rc.min())

    '''
    print("\n" + "=" * 70)
    print("ERC WEIGHTS (from equal-weight daily returns)")
    print("=" * 70)
    for k, v in erc_w.items():
        print(f"{k}: {v:.4f}")
    print("Sum:", sum(erc_w.values()))

    
    # ==========================
    # RUN PORTFOLIO WITH ERC WEIGHTS
    # ==========================

    port_summary_erc, port_eq_erc, port_log_erc = simulate_portfolio_equal_risk(
        portfolio_trades,
        market_cfg,
        start_capital=50_000.0,
        max_one_position_per_market=True,
        exposure_scale=1.0,
        weights=erc_w,
    )

    dd_summary_erc, mtm_df_erc = compute_portfolio_mtm_equity_and_intraday_dd(
        port_log_erc,
        market_cfg,
        start_capital=50_000.0,
        freq="5min",
        price_col="close",
    )


    port_summary_erc.update(dd_summary_erc)

    print("\n" + "=" * 70)
    print("PORTFÖLJ-RESULTAT (ERC weights)")
    print("=" * 70)
    for k, v in port_summary_erc.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    '''

    plt.figure(figsize=(12, 5))
    plt.plot(mtm_df.index, mtm_df["Equity_MTM"], label="Equal Weight")
    #plt.plot(mtm_df_erc.index, mtm_df_erc["Equity_MTM"], label="ERC Weight")
    #plt.title("MTM Equity Comparison: Equal vs ERC (5m)")
    plt.title("MTM Equity Equal Weights")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    print("\nDaily return std per market:")
    print(daily_ret.std())

    print("\nDaily return correlation:")
    print(daily_ret.corr())


    def block_bootstrap_1d_returns(
            ret: pd.Series,
            block_size_days: int = 20,
            n_sims: int = 50000,
            seed: int = 42,
            ann_factor: float = 365.0
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x = ret.values.astype(float)
        T = len(x)
        if T < block_size_days:
            raise ValueError("För få dagar för block_size_days")

        max_start = T - block_size_days
        n_blocks = int(np.ceil(T / block_size_days))

        out = np.empty((n_sims, 4), dtype=float)  # TotalReturn, CAGR, Sharpe, MaxDD

        for s in range(n_sims):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            sample = np.concatenate([x[i:i + block_size_days] for i in starts])[:T]

            eq = np.cumprod(1.0 + sample)
            total_return = eq[-1] - 1.0

            mu = sample.mean()
            sd = sample.std(ddof=1)
            sharpe = (mu / sd) * np.sqrt(ann_factor) if sd > 0 else np.nan

            years = T / ann_factor
            cagr = (eq[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan

            roll_max = np.maximum.accumulate(eq)
            dd = eq / roll_max - 1.0
            maxdd = dd.min()

            out[s, 0] = total_return * 100.0
            out[s, 1] = cagr * 100.0
            out[s, 2] = sharpe
            out[s, 3] = maxdd * 100.0

        return pd.DataFrame(out, columns=["TotalReturn_%", "CAGR_%", "Sharpe", "MaxDD_%"])

    #Equal Weights Bootstrapping
    def bootstrap_from_mtm_df(
            mtm_df_in: pd.DataFrame,
            label: str,
            start_capital: float = 50_000.0,
            block_size_days: int = 20,
            n_sims: int = 50000,
            seed: int = 42
    ) -> pd.DataFrame:
        mtm_daily_eq = mtm_df_in["Equity_MTM"].resample("D").last().ffill()
        mtm_daily_ret = mtm_daily_eq.pct_change().dropna()

        boot_df = block_bootstrap_1d_returns(
            mtm_daily_ret,
            block_size_days=block_size_days,
            n_sims=n_sims,
            seed=seed,
        )

        boot_df["EndEquity"] = start_capital * (1.0 + boot_df["TotalReturn_%"] / 100.0)

        print("\n" + "=" * 70)
        print(f"BOOTSTRAP SUMMARY ({label})")
        print("=" * 70)
        print(boot_df[["EndEquity", "TotalReturn_%", "MaxDD_%", "Sharpe", "CAGR_%"]]
              .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
        print("Prob(MaxDD worse than -10%):", float((boot_df["MaxDD_%"] < -10.0).mean()))
        print("Prob(MaxDD worse than -15%):", float((boot_df["MaxDD_%"] < -15.0).mean()))
        print("Prob(CAGR < 0):", float((boot_df["CAGR_%"] < 0.0).mean()))

        return boot_df

    boot_equal_df = bootstrap_from_mtm_df(
        mtm_df,  # <-- Equal weight MTM equity
        label="Equal weights",
        block_size_days=20,
        n_sims=50000,
        seed=42,
    )

    '''
    boot_erc_df = bootstrap_from_mtm_df(
        mtm_df_erc,  # <-- ERC weight MTM equity
        label="ERC weights",
        block_size_days=20,
        n_sims=50000,
        seed=42,
    )
    '''

    plt.figure(figsize=(10, 4))
    plt.hist(boot_equal_df["TotalReturn_%"], bins=60, alpha=0.5, label="Equal")
    #plt.hist(boot_erc_df["TotalReturn_%"], bins=60, alpha=0.5, label="ERC")
    plt.title("Bootstrap distribution: Total Return (%)")
    plt.xlabel("Total Return (%)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(boot_equal_df["MaxDD_%"], bins=60, alpha=0.5, label="Equal")
    #plt.hist(boot_erc_df["MaxDD_%"], bins=60, alpha=0.5, label="ERC")
    plt.title("Bootstrap distribution: Max Drawdown (%)")
    plt.xlabel("Max DD (%) (negativt)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("Inga trades att simulera i portföljen.")