#!/usr/bin/env python3
"""
Project Ares v12.2 — Goldman-Tier Macro Overlay
All 6 improvements on top of v11:
1. Partial event sizing (graduated, not binary block)
2. Regime-adaptive allocation (SPY regime shifts strategy weights)
3. VIX term structure (VIX/VIX3M ratio for entry timing)
4. Sector rotation (relative strength vs SMH)
5. Earnings momentum (post-earnings drift, beat streaks)
6. Intraday entry optimization (VWAP discount proxy)

Compares v11 vs v12.1 vs v12.2 across 4 periods × 3 strategies
"""

import json, statistics, urllib.request, math
from datetime import datetime, timedelta, date, timezone
from collections import defaultdict

SB_URL = "https://ndgyyrbyboqmfnqdnvaz.supabase.co"
SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kZ3l5cmJ5Ym9xbWZucWRudmF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA2ODY3ODksImV4cCI6MjA4NjI2Mjc4OX0.l3cc7DqKewHpr0Lufroo_lyMIMETYS4AB3PaYDArivs"
TICKERS = ['NVDA','AMD','MU','MSFT','AAPL','AMZN','GOOGL','META','AVGO','QCOM','INTC','TSM','MRVL','LRCX','AMAT']

BASE_STRATEGIES = {
    'patient': {'hold': 20, 'trail_start': 2.5, 'trail_dist': 2.5, 'stop_mult': 1.5, 'tp_pct': 0.25,
                'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
    'hybrid': {'hold': 10, 'trail_start': 1.5, 'trail_dist': 1.8, 'stop_mult': 1.2, 'tp_pct': 0.12,
               'sizing': {6: 0.20, 7: 0.35, 8: 0.55, 9: 0.90}},
    'shortterm': {'hold': 10, 'trail_start': 2.0, 'trail_dist': 2.0, 'stop_mult': 1.5, 'tp_pct': 0.15,
                  'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
}

# Fixed allocations for v11 / v12.1
FIXED_ALLOC = {'patient': 0.40, 'hybrid': 0.35, 'shortterm': 0.25}

PERIODS = {'3mo': 63, '6mo': 126, '1yr': 252, '2yr': 504}

# ─── DATA FETCHING ────────────────────────────────────────────

def fetch_yahoo(symbol, yrange='3y'):
    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?range={yrange}&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    result = data['chart']['result'][0]
    m = {}
    for t, c in zip(result['timestamp'], result['indicators']['quote'][0]['close']):
        if c is not None:
            m[datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d')] = round(c, 2)
    return m

def fetch_sb(table, params=""):
    rows = []
    offset = 0
    while True:
        url = f"{SB_URL}/rest/v1/{table}?{params}&limit=1000&offset={offset}"
        req = urllib.request.Request(url, headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
        with urllib.request.urlopen(req) as r:
            batch = json.loads(r.read())
        rows.extend(batch)
        if len(batch) < 1000: break
        offset += 1000
    return rows

def fetch_earnings():
    """Fetch earnings history from Supabase, return dict of ticker -> list of {date, surprise_pct}"""
    data = fetch_sb('earnings_history', 'order=date.asc')
    earnings = defaultdict(list)
    for row in data:
        earnings[row['ticker']].append({
            'date': row['date'],
            'surprise_pct': float(row.get('surprise_pct', 0) or 0),
        })
    return dict(earnings)

# ─── IMPROVEMENT #1: PARTIAL EVENT SIZING ─────────────────────

# CPI dates (approx mid-month)
EVENT_DATES = {}  # date_str -> 'cpi' or 'fomc'
for year in range(2020, 2027):
    for month in range(1, 13):
        for day in [10, 11, 12, 13, 14, 15]:
            try:
                d = date(year, month, day)
                if d.weekday() < 5:
                    EVENT_DATES[d.isoformat()] = 'cpi'
                    break
            except ValueError:
                continue

FOMC = [
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16','2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15','2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12','2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-11-05','2025-12-17',
    '2026-01-28','2026-03-18','2026-05-06','2026-06-17','2026-07-29','2026-09-16','2026-10-28','2026-12-09',
]
for fd in FOMC:
    EVENT_DATES[fd] = 'fomc'

def get_event_sizing(date_str):
    """Graduated event sizing instead of binary block.
    Day-of event: 0% (full block)
    1 day before: 50%
    2 days before: 75%
    Otherwise: 100%
    """
    d = date.fromisoformat(date_str)
    # Check if today IS an event
    if date_str in EVENT_DATES:
        return 0.0
    # Check if tomorrow is an event
    for offset in range(1, 3):
        check = d + timedelta(days=offset)
        # Skip weekends
        while check.weekday() >= 5:
            check += timedelta(days=1)
        if check.isoformat() in EVENT_DATES:
            if offset == 1:
                return 0.50  # day before
            else:
                return 0.75  # 2 days before
    return 1.0

# ─── IMPROVEMENT #2: REGIME-ADAPTIVE ALLOCATION ───────────────

def get_spy_regime(spy_map, date_str):
    """Determine market regime from SPY price action.
    Returns allocation dict for the 3 strategies.
    """
    d = date.fromisoformat(date_str)
    # Get SPY values for SMA calculation
    spy_vals = []
    for i in range(60):
        check = (d - timedelta(days=i)).isoformat()
        if check in spy_map:
            spy_vals.append(spy_map[check])
    
    if len(spy_vals) < 50:
        return FIXED_ALLOC, 'unknown'
    
    current = spy_vals[0]
    sma20 = statistics.mean(spy_vals[:20]) if len(spy_vals) >= 20 else current
    sma50 = statistics.mean(spy_vals[:50]) if len(spy_vals) >= 50 else current
    
    if current > sma20 > sma50:
        # Strong bull — favor short-term (higher win rate in trends)
        return {'patient': 0.25, 'hybrid': 0.35, 'shortterm': 0.40}, 'bull'
    elif current < sma20 and current < sma50:
        # Bear — favor patient (wider stops survive chop)
        return {'patient': 0.45, 'hybrid': 0.35, 'shortterm': 0.20}, 'bear'
    else:
        # Transition — equal weight
        return {'patient': 0.33, 'hybrid': 0.34, 'shortterm': 0.33}, 'transition'

# ─── IMPROVEMENT #3: VIX TERM STRUCTURE ───────────────────────

def get_vix_term_mult(vix_val, vix3m_val):
    """VIX/VIX3M ratio signals panic vs structural fear.
    Ratio > 1.1 = backwardation = panic spike (contrarian buy)
    Ratio 0.9-1.1 = normal
    Ratio < 0.85 = contango = prolonged fear (reduce)
    """
    if vix_val is None or vix3m_val is None or vix3m_val == 0:
        return 1.0
    ratio = vix_val / vix3m_val
    if ratio > 1.15:
        return 1.25  # Strong contrarian signal — panic is temporary
    elif ratio > 1.05:
        return 1.10  # Mild backwardation
    elif ratio < 0.82:
        return 0.6   # Deep contango — market expects prolonged pain
    elif ratio < 0.90:
        return 0.8   # Mild contango
    return 1.0

# ─── IMPROVEMENT #4: SECTOR ROTATION ─────────────────────────

def get_relative_strength(ticker_closes, smh_map, date_str, lookback=20):
    """Compare stock's recent performance vs SMH.
    Outperformers get boosted, underperformers get reduced.
    """
    d = date.fromisoformat(date_str)
    smh_vals = []
    for i in range(lookback + 1):
        check = (d - timedelta(days=i)).isoformat()
        if check in smh_map:
            smh_vals.append(smh_map[check])
    
    if len(smh_vals) < 10 or len(ticker_closes) < lookback:
        return 1.0
    
    smh_ret = (smh_vals[0] - smh_vals[-1]) / smh_vals[-1] * 100
    stock_ret = (ticker_closes[-1] - ticker_closes[-lookback]) / ticker_closes[-lookback] * 100
    
    relative = stock_ret - smh_ret
    if relative > 3:
        return 1.15  # Outperforming sector — relative strength
    elif relative < -5:
        return 0.7   # Underperforming badly — might be broken
    elif relative < -2:
        return 0.85  # Slightly weak
    return 1.0

# ─── IMPROVEMENT #5: EARNINGS MOMENTUM ───────────────────────

def get_earnings_mult(ticker, date_str, earnings_data):
    """Earnings-based sizing adjustment.
    - Post-earnings drift: 60 days after beat > 10% = boost
    - Pre-earnings caution: 14 days before = reduce
    - Beat streak (4+ consecutive) = boost conviction
    """
    if ticker not in earnings_data:
        return 1.0, False
    
    d = date.fromisoformat(date_str)
    earns = earnings_data[ticker]
    
    # Check pre-earnings (14 days before next earnings)
    for e in earns:
        ed = date.fromisoformat(e['date'])
        days_until = (ed - d).days
        if 0 < days_until <= 14:
            return 0.5, True  # Reduce before earnings
    
    # Check post-earnings drift
    recent_beats = []
    for e in reversed(earns):
        ed = date.fromisoformat(e['date'])
        days_since = (d - ed).days
        if days_since < 0:
            continue
        if days_since <= 60 and e['surprise_pct'] > 10:
            return 1.2, False  # Post-earnings drift boost
        if days_since <= 365:
            recent_beats.append(e['surprise_pct'] > 0)
        if len(recent_beats) >= 4:
            break
    
    # Beat streak check
    if len(recent_beats) >= 4 and all(recent_beats[:4]):
        return 1.1, False  # 4-quarter beat streak
    
    return 1.0, False

# ─── IMPROVEMENT #6: INTRADAY ENTRY OPTIMIZATION ─────────────

def get_entry_discount(atr, price):
    """Simulate VWAP/limit order entry vs market order.
    Assume we can get ~0.3% better entry on average by working the order.
    More volatile stocks = bigger potential discount.
    """
    atr_pct = atr / price
    # Higher ATR = wider intraday range = more room for limit fills
    discount = min(0.005, atr_pct * 0.15)  # Cap at 0.5%
    return 1.0 - discount  # Price multiplier (e.g., 0.997 = 0.3% cheaper)

# ─── CORE ENGINE (from v11) ──────────────────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-period-1+i] - closes[-period-1+i-1]
        gains.append(max(0, d)); losses.append(max(0, -d))
    ag, al = statistics.mean(gains), statistics.mean(losses)
    if al == 0: return 100
    return 100 - 100 / (1 + ag / al)

def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(-period, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return statistics.mean(trs)

def get_vol_tier(atr_pct):
    if atr_pct > 2.5: return 'high'
    elif atr_pct > 1.5: return 'mid'
    return 'low'

def get_adaptive_params(vol_tier, base_strat):
    s = dict(base_strat)
    if vol_tier == 'high':
        s['stop_mult'] = max(s['stop_mult'], 2.5); s['trail_dist'] = max(s['trail_dist'], 3.0); s['trail_start'] = max(s['trail_start'], 3.0)
    elif vol_tier == 'low':
        s['stop_mult'] = min(s['stop_mult'], 1.2); s['tp_pct'] = min(s['tp_pct'], 0.12)
        s['trail_start'] = min(s['trail_start'], 1.5); s['trail_dist'] = min(s['trail_dist'], 1.5)
    return s

def check_recovery_pattern(closes, idx, lookback=250):
    if idx < lookback: return None
    bounces, total_dips = 0, 0
    for i in range(max(15, idx - lookback), idx - 10, 5):
        sub = closes[i-14:i+1]
        if len(sub) < 15: continue
        rsi = calc_rsi(sub)
        if rsi < 35:
            total_dips += 1
            entry = closes[i]
            mx = max(closes[i+1:i+11]) if i + 11 <= len(closes) else entry
            if (mx - entry) / entry * 100 > 5: bounces += 1
    if total_dips < 3: return None
    return bounces / total_dips >= 0.6

TICKER_DATA = {
    'NVDA': {'peg': 0.82, 'beat_rate': 0.90}, 'AMD': {'peg': 1.1, 'beat_rate': 0.80},
    'MU': {'peg': 0.3, 'beat_rate': 0.75}, 'MSFT': {'peg': 1.8, 'beat_rate': 0.85},
    'AAPL': {'peg': 2.0, 'beat_rate': 0.80}, 'AMZN': {'peg': 1.2, 'beat_rate': 0.80},
    'GOOGL': {'peg': 1.1, 'beat_rate': 0.80}, 'META': {'peg': 0.9, 'beat_rate': 0.85},
    'AVGO': {'peg': 1.0, 'beat_rate': 0.85}, 'QCOM': {'peg': 1.3, 'beat_rate': 0.75},
    'INTC': {'peg': 2.5, 'beat_rate': 0.50}, 'TSM': {'peg': 0.8, 'beat_rate': 0.80},
    'MRVL': {'peg': 1.2, 'beat_rate': 0.75}, 'LRCX': {'peg': 1.1, 'beat_rate': 0.80},
    'AMAT': {'peg': 0.9, 'beat_rate': 0.85},
}

def calc_confluence(closes, highs, lows, volumes, idx, ticker_data):
    if idx < 200: return 0, 50, 1, 1, 0, 0
    price = closes[idx]
    sma50 = statistics.mean(closes[idx-50:idx])
    sma200 = statistics.mean(closes[idx-200:idx])
    rsi = calc_rsi(closes[:idx+1])
    roc5 = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
    avg_vol = statistics.mean(volumes[idx-20:idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    atr = calc_atr(highs[:idx+1], lows[:idx+1], closes[:idx+1])
    atr_pct = atr / price * 100
    votes = 0
    if rsi <= 35: votes += 1
    if -8 < (price - sma50) / sma50 * 100 < 0: votes += 1
    if price > sma200: votes += 1
    if -3 < roc5 < 3: votes += 1
    if vol_ratio > 1.3 and rsi < 45: votes += 1
    if ticker_data.get('peg', 1.5) < 1.5: votes += 1
    if ticker_data.get('beat_rate', 0.75) >= 0.75: votes += 1
    low_52 = min(closes[max(0, idx-252):idx+1]); high_52 = max(closes[max(0, idx-252):idx+1])
    if high_52 > low_52 and 0.2 <= (price - low_52) / (high_52 - low_52) <= 0.6: votes += 1
    if atr_pct < 4: votes += 1
    if check_recovery_pattern(closes, idx): votes += 1
    return votes, rsi, atr, atr_pct, sma50, sma200

# ─── VIX HELPERS ──────────────────────────────────────────────

def get_vix_regime(vix_val):
    if vix_val is None: return 'normal', 1.0
    if vix_val >= 30: return 'crisis', 0.0
    elif vix_val >= 25: return 'elevated', 0.5
    return 'normal', 1.0

def is_vix_declining(vix_map, date_str, lookback=5):
    d = date.fromisoformat(date_str)
    vals = []
    for i in range(lookback + 1):
        check = (d - timedelta(days=i)).isoformat()
        if check in vix_map: vals.append(vix_map[check])
    if len(vals) < 3: return False
    return vals[0] < vals[-1]

# v12.1 event blocking (binary)
V121_EVENTS = set()
for ds in EVENT_DATES:
    V121_EVENTS.add(ds)
    d = date.fromisoformat(ds)
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5: prev -= timedelta(days=1)
    V121_EVENTS.add(prev.isoformat())

# ─── BACKTEST ENGINE ─────────────────────────────────────────

def run_strat(version, vix_map, vix3m_map, spy_map, smh_map, earnings_data, all_data, sample, bs, td, sname, scfg):
    alloc = FIXED_ALLOC[sname]
    capital = 10000 * alloc
    cash = capital
    positions = {}
    trades = []
    peak = capital; max_dd = 0

    for di in range(bs, td):
        cd = all_data[sample]['dates'][di]
        vix = vix_map.get(cd)
        vix3m = vix3m_map.get(cd)

        # v12.2: Regime-adaptive allocation
        if version == 'v12.2':
            regime_alloc, regime = get_spy_regime(spy_map, cd)
            alloc = regime_alloc[sname]
            # Recalculate effective capital for this strategy
            eff_capital = 10000 * alloc
        else:
            eff_capital = capital

        # Portfolio value
        pv = cash
        for t, p in positions.items():
            if t in all_data and di < len(all_data[t]['closes']):
                pv += p['sh'] * all_data[t]['closes'][di]
        if pv > peak: peak = pv
        dd = (peak - pv) / peak * 100
        if dd > max_dd: max_dd = dd

        # Exits
        for t in list(positions.keys()):
            if t not in all_data or di >= len(all_data[t]['closes']): continue
            p = positions[t]; price = all_data[t]['closes'][di]
            if price > p['th']: p['th'] = price
            if p['th'] >= p['ta']:
                nt = p['th'] - p['td']
                if p['ts'] is None or nt > p['ts']: p['ts'] = nt
            er = None
            if price <= p['sl']: er = 'STOP'
            elif p['ts'] and price <= p['ts']: er = 'TRAIL'
            elif (price - p['ep']) / p['ep'] >= p['tp']: er = 'TP'
            elif di - p['ei'] >= p['mh']: er = 'MH'
            if er:
                pnl = (price - p['ep']) / p['ep'] * 100
                dollar = (price - p['ep']) * p['sh']
                trades.append({'pnl': pnl, 'dollar': dollar, 'reason': er})
                cash += p['sh'] * price
                del positions[t]

        # Entries
        for t in TICKERS:
            if t in positions or t not in all_data or di >= len(all_data[t]['closes']): continue
            d = all_data[t]; price = d['closes'][di]
            r = calc_confluence(d['closes'], d['highs'], d['lows'], d['volumes'], di, TICKER_DATA.get(t, {}))
            votes, rsi, atr, atr_pct, sma50, sma200 = r
            if votes < 6 or (rsi > 50 and price > sma50): continue

            macro_mult = 1.0

            if version == 'v12.1':
                # Binary event block
                if cd in V121_EVENTS: continue
                if vix is not None:
                    reg, rm = get_vix_regime(vix)
                    if reg == 'crisis':
                        if not is_vix_declining(vix_map, cd): continue
                        else: rm = 0.5
                    macro_mult *= rm
                    # Contrarian boost
                    if vix >= 22 and is_vix_declining(vix_map, cd):
                        macro_mult *= 1.2

            elif version == 'v12.2':
                # #1: Partial event sizing
                evt_mult = get_event_sizing(cd)
                if evt_mult == 0: continue
                macro_mult *= evt_mult

                # VIX regime (same as v12.1)
                if vix is not None:
                    reg, rm = get_vix_regime(vix)
                    if reg == 'crisis':
                        if not is_vix_declining(vix_map, cd): continue
                        else: rm = 0.5
                    macro_mult *= rm

                # #3: VIX term structure
                macro_mult *= get_vix_term_mult(vix, vix3m)

                # #4: Sector rotation
                rs_mult = get_relative_strength(d['closes'][:di+1], smh_map, cd)
                macro_mult *= rs_mult

                # #5: Earnings momentum
                earn_mult, pre_earnings = get_earnings_mult(t, cd, earnings_data)
                macro_mult *= earn_mult

                # #6: Entry discount
                entry_discount = get_entry_discount(atr, price)
                price = price * entry_discount  # Simulated better fill

            sz = scfg['sizing'].get(min(votes, 9), scfg['sizing'][6])
            pos_value = eff_capital * sz * macro_mult
            if pos_value > cash or pos_value < 50: continue
            sh = int(pos_value / price)
            if sh < 1: continue

            vt = get_vol_tier(atr_pct)
            ad = get_adaptive_params(vt, scfg)
            positions[t] = {
                'ep': price, 'ei': di, 'sh': sh,
                'sl': price - ad['stop_mult'] * atr,
                'ta': price + ad['trail_start'] * atr,
                'td': ad['trail_dist'] * atr, 'th': price, 'ts': None,
                'tp': ad['tp_pct'], 'mh': ad['hold'],
            }
            cash -= sh * price

    # Close remaining
    for t in list(positions.keys()):
        p = positions[t]; price = all_data[t]['closes'][-1]
        pnl = (price - p['ep']) / p['ep'] * 100
        cash += p['sh'] * price
        trades.append({'pnl': pnl, 'dollar': (price - p['ep']) * p['sh'], 'reason': 'OPEN'})

    ret = (cash - capital) / capital * 100
    wins = [x for x in trades if x['pnl'] > 0]
    losses = [x for x in trades if x['pnl'] <= 0]
    wp = len(wins) / max(len(trades), 1) * 100
    tw = sum(x['dollar'] for x in wins) if wins else 0
    tl = abs(sum(x['dollar'] for x in losses)) if losses else 0.01
    pf = tw / tl
    avg_w = statistics.mean([x['pnl'] for x in wins]) if wins else 0
    avg_l = statistics.mean([x['pnl'] for x in losses]) if losses else 0
    return {'ret': ret, 'n': len(trades), 'wp': wp, 'pf': pf, 'dd': max_dd, 'aw': avg_w, 'al': avg_l}


def main():
    print("█" * 70)
    print("  ARES v11 vs v12.1 vs v12.2 (GOLDMAN) — MULTI-PERIOD")
    print("█" * 70)

    print("\n📊 Fetching data...")
    vix_map = fetch_yahoo('^VIX', '6y')
    vix3m_map = fetch_yahoo('^VIX3M', '3y')
    spy_map = fetch_yahoo('SPY', '3y')
    smh_map = fetch_yahoo('SMH', '3y')
    print(f"  VIX:{len(vix_map)}d VIX3M:{len(vix3m_map)}d SPY:{len(spy_map)}d SMH:{len(smh_map)}d")

    earnings_data = fetch_earnings()
    print(f"  Earnings: {sum(len(v) for v in earnings_data.values())} records for {len(earnings_data)} tickers")

    cutoff = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    all_data = {}
    for ticker in TICKERS:
        data = fetch_sb('daily_prices', f'ticker=eq.{ticker}&date=gte.{cutoff}&order=date.asc')
        if len(data) < 250: continue
        all_data[ticker] = {
            'closes': [float(d['close']) for d in data], 'highs': [float(d['high']) for d in data],
            'lows': [float(d['low']) for d in data], 'volumes': [int(d['volume']) for d in data],
            'dates': [d['date'] for d in data],
        }
    print(f"  Stocks: {len(all_data)} tickers loaded")

    sample = list(all_data.keys())[0]
    td = len(all_data[sample]['dates'])

    VERSIONS = ['v11', 'v12.1', 'v12.2']

    for sname, scfg in BASE_STRATEGIES.items():
        print(f"\n{'='*70}")
        print(f"  {sname.upper()} STRATEGY")
        print(f"{'='*70}")
        print(f"  {'Period':<6s} | {'Ver':<5s} | {'Return':>8s} | {'Trades':>6s} | {'Win%':>5s} | {'PF':>5s} | {'MaxDD':>5s} | {'AvgW':>6s} | {'AvgL':>6s}")
        print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}")

        for pname, pdays in PERIODS.items():
            bs = max(200, td - pdays)
            if bs >= td - 20: continue
            results = {}
            for ver in VERSIONS:
                r = run_strat(ver, vix_map, vix3m_map, spy_map, smh_map, earnings_data, all_data, sample, bs, td, sname, scfg)
                results[ver] = r
                print(f"  {pname if ver=='v11' else '':6s} | {ver:<5s} | {r['ret']:>+7.1f}% | {r['n']:>6d} | {r['wp']:>4.1f}% | {r['pf']:>5.2f} | {r['dd']:>4.1f}% | {r['aw']:>+5.1f}% | {r['al']:>+5.1f}%")

            # Delta v12.2 vs v11
            v11 = results['v11']; v122 = results['v12.2']
            dr = v122['ret'] - v11['ret']; ddd = v122['dd'] - v11['dd']; dpf = v122['pf'] - v11['pf']
            ri = '✅' if dr >= 0 else '❌'; di = '✅' if ddd <= 0 else '❌'; pi = '✅' if dpf >= 0 else '❌'
            print(f"  {'Δ12.2':6s} |       | {dr:>+7.1f}%{ri}|        |       | {dpf:>+4.2f}{pi}| {ddd:>+4.1f}%{di}|")
            print(f"  {'─'*6}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*6}")

    # Overall summary
    print(f"\n{'█'*70}")
    print("  OVERALL WINNER BY PERIOD")
    print(f"{'█'*70}")
    for pname, pdays in PERIODS.items():
        bs = max(200, td - pdays)
        if bs >= td - 20: continue
        totals = {v: 0 for v in VERSIONS}
        for sname, scfg in BASE_STRATEGIES.items():
            for ver in VERSIONS:
                r = run_strat(ver, vix_map, vix3m_map, spy_map, smh_map, earnings_data, all_data, sample, bs, td, sname, scfg)
                totals[ver] += r['ret'] * FIXED_ALLOC[sname]
        best = max(totals, key=totals.get)
        print(f"  {pname}: v11={totals['v11']:>+6.1f}% | v12.1={totals['v12.1']:>+6.1f}% | v12.2={totals['v12.2']:>+6.1f}% → 🏆 {best}")

    print(f"{'█'*70}")

if __name__ == '__main__':
    main()
