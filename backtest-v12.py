#!/usr/bin/env python3
"""
Project Ares v12 Backtest — Macro Overlay on v11
Adds: VIX regime filter, Fear & Greed position sizing, Event calendar blocking
Run against full data range (training + validation) from Supabase
"""

import json, sys, statistics, urllib.request
from datetime import datetime, timedelta, date
from collections import defaultdict

SB_URL = "https://ndgyyrbyboqmfnqdnvaz.supabase.co"
SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kZ3l5cmJ5Ym9xbWZucWRudmF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA2ODY3ODksImV4cCI6MjA4NjI2Mjc4OX0.l3cc7DqKewHpr0Lufroo_lyMIMETYS4AB3PaYDArivs"

TICKERS = ['NVDA','AMD','MU','MSFT','AAPL','AMZN','GOOGL','META','AVGO','QCOM','INTC','TSM','MRVL','LRCX','AMAT']

STRATEGIES = {
    'patient': {'hold': 20, 'trail_start': 2.5, 'trail_dist': 2.5, 'stop_mult': 1.5, 'tp_pct': 0.25, 'alloc': 0.40,
                'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
    'hybrid': {'hold': 10, 'trail_start': 1.5, 'trail_dist': 1.8, 'stop_mult': 1.2, 'tp_pct': 0.12, 'alloc': 0.35,
               'sizing': {6: 0.20, 7: 0.35, 8: 0.55, 9: 0.90}},
    'shortterm': {'hold': 10, 'trail_start': 2.0, 'trail_dist': 2.0, 'stop_mult': 1.5, 'tp_pct': 0.15, 'alloc': 0.25,
                  'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
}

# ─── MACRO DATA ──────────────────────────────────────────────

def fetch_vix_data():
    """Fetch VIX historical from Yahoo Finance"""
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/%5EVIX?range=6y&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']
    vix_map = {}
    for t, c in zip(timestamps, closes):
        if c is not None:
            d = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
            vix_map[d] = round(c, 2)
    return vix_map

def fetch_spy_data():
    """Fetch SPY for market regime"""
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=6y&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']
    spy_map = {}
    for t, c in zip(timestamps, closes):
        if c is not None:
            d = datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
            spy_map[d] = round(c, 2)
    return spy_map

# Historical high-impact economic event dates (CPI, NFP, FOMC, GDP, PCE)
# These are dates where we BLOCK new entries (24h before = the day before + day of)
HIGH_IMPACT_EVENTS = set()

# CPI releases (approximate — 2nd/3rd week of month)
for year in range(2020, 2027):
    # CPI typically mid-month
    for month in range(1, 13):
        # Approximate CPI dates (10th-15th of each month)
        for day in [10, 11, 12, 13, 14, 15]:
            try:
                d = date(year, month, day)
                if d.weekday() < 5:  # weekday
                    HIGH_IMPACT_EVENTS.add(d.isoformat())
                    # Also block day before
                    prev = d - timedelta(days=1)
                    if prev.weekday() >= 5:
                        prev = prev - timedelta(days=(prev.weekday() - 4))
                    HIGH_IMPACT_EVENTS.add(prev.isoformat())
                    break
            except ValueError:
                continue

# NFP (first Friday of each month)
for year in range(2020, 2027):
    for month in range(1, 13):
        d = date(year, month, 1)
        while d.weekday() != 4:  # Friday
            d += timedelta(days=1)
        HIGH_IMPACT_EVENTS.add(d.isoformat())
        prev = d - timedelta(days=1)
        HIGH_IMPACT_EVENTS.add(prev.isoformat())

# FOMC meeting dates (hardcoded known dates — 8 per year)
FOMC_DATES = [
    # 2020
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    # 2021
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16','2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    # 2022
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15','2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    # 2023
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    # 2024
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12','2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    # 2025
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-11-05','2025-12-17',
    # 2026
    '2026-01-28','2026-03-18','2026-05-06','2026-06-17','2026-07-29','2026-09-16','2026-10-28','2026-12-09',
]
for fd in FOMC_DATES:
    HIGH_IMPACT_EVENTS.add(fd)
    d = date.fromisoformat(fd)
    prev = d - timedelta(days=1)
    if prev.weekday() >= 5:
        prev = prev - timedelta(days=(prev.weekday() - 4))
    HIGH_IMPACT_EVENTS.add(prev.isoformat())

def is_event_blocked(date_str):
    """Check if a date is within event blocking window"""
    return date_str in HIGH_IMPACT_EVENTS

def get_vix_regime(vix_val):
    """VIX regime classification"""
    if vix_val is None:
        return 'normal', 1.0
    if vix_val >= 30:
        return 'crisis', 0.0      # NO new entries
    elif vix_val >= 25:
        return 'elevated', 0.5    # Half size
    elif vix_val >= 20:
        return 'caution', 0.8     # Slight reduction
    else:
        return 'normal', 1.0      # Full size

def get_fg_multiplier(vix_val):
    """
    Use VIX as proxy for Fear & Greed (inverted relationship).
    VIX < 15 ≈ Extreme Greed (reduce), VIX 15-20 ≈ Greed/Neutral (normal),
    VIX 20-30 ≈ Fear (contrarian opportunity), VIX > 30 = blocked above
    """
    if vix_val is None:
        return 1.0
    if vix_val < 13:
        return 0.5    # Extreme complacency → reduce size
    elif vix_val < 16:
        return 0.8    # Low vol → slight reduction (potential complacency)
    elif vix_val < 22:
        return 1.0    # Normal range
    elif vix_val < 28:
        return 1.0    # Fear zone — contrarian, normal size (opportunity)
    else:
        return 0.5    # Extreme fear — reduce (or blocked by regime)

def is_vix_declining(vix_map, date_str, lookback=5):
    """Check if VIX is declining over last N days"""
    d = date.fromisoformat(date_str)
    vals = []
    for i in range(lookback + 1):
        check = (d - timedelta(days=i)).isoformat()
        if check in vix_map:
            vals.append(vix_map[check])
    if len(vals) < 3:
        return False
    return vals[0] < vals[-1]  # current < lookback days ago

# ─── CORE ENGINE (from v11) ──────────────────────────────────

def fetch(table, params=""):
    """Fetch with pagination to handle large datasets"""
    all_rows = []
    offset = 0
    batch = 1000
    while True:
        url = f"{SB_URL}/rest/v1/{table}?{params}&limit={batch}&offset={offset}"
        req = urllib.request.Request(url, headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
        with urllib.request.urlopen(req) as r:
            rows = json.loads(r.read())
        all_rows.extend(rows)
        if len(rows) < batch:
            break
        offset += batch
    return all_rows

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-period-1+i] - closes[-period-1+i-1]
        gains.append(max(0, d))
        losses.append(max(0, -d))
    avg_gain = statistics.mean(gains)
    avg_loss = statistics.mean(losses)
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(-period, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return statistics.mean(trs)

def get_vol_tier(atr_pct):
    if atr_pct > 2.5: return 'high'
    elif atr_pct > 1.5: return 'mid'
    else: return 'low'

def get_adaptive_params(vol_tier, base_strat):
    s = dict(base_strat)
    if vol_tier == 'high':
        s['stop_mult'] = max(s['stop_mult'], 2.5)
        s['trail_dist'] = max(s['trail_dist'], 3.0)
        s['trail_start'] = max(s['trail_start'], 3.0)
    elif vol_tier == 'low':
        s['stop_mult'] = min(s['stop_mult'], 1.2)
        s['tp_pct'] = min(s['tp_pct'], 0.12)
        s['trail_start'] = min(s['trail_start'], 1.5)
        s['trail_dist'] = min(s['trail_dist'], 1.5)
    return s

def check_recovery_pattern(closes, idx, lookback=250):
    """Simplified V-bounce check — sample every 5th day to reduce computation"""
    if idx < lookback: return None
    bounces, total_dips = 0, 0
    for i in range(max(15, idx - lookback), idx - 10, 5):  # Step by 5
        sub = closes[i-14:i+1]
        if len(sub) < 15: continue
        rsi = calc_rsi(sub)
        if rsi < 35:
            total_dips += 1
            entry = closes[i]
            max_next_10 = max(closes[i+1:i+11]) if i + 11 <= len(closes) else entry
            if (max_next_10 - entry) / entry * 100 > 5:
                bounces += 1
    if total_dips < 3: return None
    return bounces / total_dips >= 0.6

def calc_confluence(closes, highs, lows, volumes, idx, ticker_data):
    if idx < 200: return 0, [], 50, 1, 1, 0, 0
    price = closes[idx]
    sma50 = statistics.mean(closes[idx-50:idx])
    sma200 = statistics.mean(closes[idx-200:idx])
    rsi = calc_rsi(closes[:idx+1])
    roc5 = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
    avg_vol = statistics.mean(volumes[idx-20:idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    atr = calc_atr(highs[:idx+1], lows[:idx+1], closes[:idx+1])
    atr_pct = atr / price * 100
    
    votes, factors = 0, []
    if rsi <= 35: votes += 1; factors.append('RSI≤35')
    d50 = (price - sma50) / sma50 * 100
    if -8 < d50 < 0: votes += 1; factors.append('Near SMA50')
    if price > sma200: votes += 1; factors.append('Above SMA200')
    if -3 < roc5 < 3: votes += 1; factors.append('Stabilizing')
    if vol_ratio > 1.3 and rsi < 45: votes += 1; factors.append('Vol+Oversold')
    if ticker_data.get('peg', 1.5) < 1.5: votes += 1; factors.append('PEG<1.5')
    if ticker_data.get('beat_rate', 0.75) >= 0.75: votes += 1; factors.append('Beats≥75%')
    low_52 = min(closes[max(0, idx-252):idx+1])
    high_52 = max(closes[max(0, idx-252):idx+1])
    if high_52 > low_52:
        pos = (price - low_52) / (high_52 - low_52)
        if 0.2 <= pos <= 0.6: votes += 1; factors.append('52w Sweet')
    if atr_pct < 4: votes += 1; factors.append('Low Vol')
    is_bouncer = check_recovery_pattern(closes, idx)
    if is_bouncer: votes += 1; factors.append('V-Bounce ✓')
    
    return votes, factors, rsi, atr, atr_pct, sma50, sma200

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

# ─── MAIN BACKTEST ───────────────────────────────────────────

def run_backtest(version, vix_map, spy_map, all_data, backtest_start, total_days):
    """Run backtest for given version"""
    
    sample_ticker = list(all_data.keys())[0]
    start_date = all_data[sample_ticker]['dates'][backtest_start]
    end_date = all_data[sample_ticker]['dates'][-1]
    
    print(f"\n{'='*70}")
    print(f"PROJECT ARES {version} BACKTEST")
    print(f"Period: {start_date} → {end_date} ({total_days - backtest_start} trading days)")
    print(f"{'='*70}")
    
    if version == 'v12':
        print(f"  Macro overlay: VIX regime + F&G sizing + Event blocking")
    
    STARTING_CAPITAL = 10000
    combined_results = {}
    
    for strat_name, strat_config in STRATEGIES.items():
        capital = STARTING_CAPITAL * strat_config['alloc']
        cash = capital
        positions = {}
        trades = []
        ticker_pnl = defaultdict(float)
        ticker_trades = defaultdict(list)
        blocked_entries = 0
        vix_reduced = 0
        
        # Track daily portfolio value for drawdown calc
        daily_values = []
        peak_value = capital
        max_drawdown = 0
        
        for day_idx in range(backtest_start, total_days):
            current_date = all_data[sample_ticker]['dates'][day_idx]
            
            # Get VIX for today
            vix_today = vix_map.get(current_date)
            
            # Calculate portfolio value
            port_value = cash
            for ticker, pos in positions.items():
                if ticker in all_data and day_idx < len(all_data[ticker]['closes']):
                    port_value += pos['shares'] * all_data[ticker]['closes'][day_idx]
            daily_values.append(port_value)
            if port_value > peak_value:
                peak_value = port_value
            dd = (peak_value - port_value) / peak_value * 100
            if dd > max_drawdown:
                max_drawdown = dd
            
            # Check exits first
            for ticker in list(positions.keys()):
                if ticker not in all_data: continue
                d = all_data[ticker]
                if day_idx >= len(d['closes']): continue
                pos = positions[ticker]
                price = d['closes'][day_idx]
                
                if price > pos['trail_high']:
                    pos['trail_high'] = price
                
                if pos['trail_high'] >= pos['trail_activation']:
                    new_trail = pos['trail_high'] - pos['trail_dist_abs']
                    if pos['trail_stop'] is None or new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail
                
                exit_reason = None
                exit_price = price
                
                if price <= pos['stop_loss']: exit_reason = 'STOP'
                elif pos['trail_stop'] and price <= pos['trail_stop']: exit_reason = 'TRAIL'
                elif (price - pos['entry_price']) / pos['entry_price'] >= pos['tp_pct']: exit_reason = 'TP'
                elif day_idx - pos['entry_idx'] >= pos['max_hold']: exit_reason = 'MAX_HOLD'
                
                if exit_reason:
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                    pnl_dollar = (exit_price - pos['entry_price']) * pos['shares']
                    cash += pos['shares'] * exit_price
                    trade = {
                        'ticker': ticker, 'entry_date': d['dates'][pos['entry_idx']],
                        'exit_date': d['dates'][day_idx], 'entry_price': pos['entry_price'],
                        'exit_price': exit_price, 'shares': pos['shares'],
                        'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar,
                        'exit_reason': exit_reason, 'confluence': pos['confluence'],
                        'vol_tier': pos.get('vol_tier', '?'), 'days': day_idx - pos['entry_idx'],
                        'macro_mult': pos.get('macro_mult', 1.0),
                    }
                    trades.append(trade)
                    ticker_pnl[ticker] += pnl_dollar
                    ticker_trades[ticker].append(trade)
                    del positions[ticker]
            
            # Check entries
            for ticker in TICKERS:
                if ticker in positions or ticker not in all_data: continue
                d = all_data[ticker]
                if day_idx >= len(d['closes']): continue
                
                price = d['closes'][day_idx]
                result = calc_confluence(d['closes'], d['highs'], d['lows'], d['volumes'],
                                        day_idx, TICKER_DATA.get(ticker, {}))
                votes, factors, rsi, atr, atr_pct, sma50, sma200 = result
                
                if votes < 6: continue
                if rsi > 50 and price > sma50: continue
                
                # ─── v12 MACRO FILTERS ───
                macro_mult = 1.0
                if version == 'v12':
                    # 1. Event calendar blocking
                    if is_event_blocked(current_date):
                        blocked_entries += 1
                        continue
                    
                    # 2. VIX regime filter
                    if vix_today is not None:
                        regime, regime_mult = get_vix_regime(vix_today)
                        if regime == 'crisis':
                            # Allow if VIX is declining from spike (mean reversion opportunity)
                            if not is_vix_declining(vix_map, current_date):
                                blocked_entries += 1
                                continue
                            else:
                                regime_mult = 0.5  # Half size on declining crisis
                        macro_mult *= regime_mult
                        
                        # 3. F&G proxy sizing (VIX-based)
                        fg_mult = get_fg_multiplier(vix_today)
                        macro_mult *= fg_mult
                        
                        if macro_mult < 1.0:
                            vix_reduced += 1
                
                # Position sizing
                sizing = strat_config['sizing']
                size_pct = sizing.get(min(votes, 9), sizing[6])
                pos_value = capital * size_pct * macro_mult
                
                if pos_value > cash or pos_value < 50: continue
                
                shares = int(pos_value / price)
                if shares < 1: continue
                
                # Adaptive stops (from v11)
                vol_tier = get_vol_tier(atr_pct)
                adapted = get_adaptive_params(vol_tier, strat_config)
                stop_mult = adapted['stop_mult']
                trail_start_mult = adapted['trail_start']
                trail_dist_mult = adapted['trail_dist']
                tp_pct = adapted['tp_pct']
                max_hold = adapted['hold']
                
                stop_loss = price - stop_mult * atr
                trail_activation = price + trail_start_mult * atr
                trail_dist_abs = trail_dist_mult * atr
                
                positions[ticker] = {
                    'entry_price': price, 'entry_idx': day_idx, 'shares': shares,
                    'stop_loss': stop_loss, 'trail_activation': trail_activation,
                    'trail_dist_abs': trail_dist_abs, 'trail_high': price,
                    'trail_stop': None, 'tp_pct': tp_pct, 'max_hold': max_hold,
                    'confluence': votes, 'vol_tier': vol_tier, 'macro_mult': macro_mult,
                }
                cash -= shares * price
        
        # Close remaining positions
        for ticker in list(positions.keys()):
            d = all_data[ticker]
            pos = positions[ticker]
            price = d['closes'][-1]
            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            pnl_dollar = (price - pos['entry_price']) * pos['shares']
            cash += pos['shares'] * price
            trades.append({
                'ticker': ticker, 'entry_price': pos['entry_price'], 'exit_price': price,
                'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar, 'exit_reason': 'OPEN',
                'confluence': pos['confluence'], 'vol_tier': pos.get('vol_tier', '?'),
                'entry_date': d['dates'][pos['entry_idx']], 'exit_date': d['dates'][-1],
                'shares': pos['shares'], 'days': len(d['closes']) - 1 - pos['entry_idx'],
                'macro_mult': pos.get('macro_mult', 1.0),
            })
            ticker_pnl[ticker] += pnl_dollar
            ticker_trades[ticker].append(trades[-1])
        
        # Results
        final_value = cash
        total_return = (final_value - capital) / capital * 100
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        
        print(f"\n{'─'*50}")
        print(f"Strategy: {strat_name.upper()} ({strat_config['alloc']*100:.0f}% = ${capital:.0f})")
        print(f"{'─'*50}")
        print(f"  ${capital:.0f} → ${final_value:.0f} ({total_return:+.1f}%)")
        print(f"  Trades: {len(trades)} | Win: {len(wins)} | Loss: {len(losses)} | Win%: {len(wins)/max(len(trades),1)*100:.1f}%")
        print(f"  Max Drawdown: {max_drawdown:.1f}%")
        
        if wins:
            print(f"  Avg win: +{statistics.mean([t['pnl_pct'] for t in wins]):.2f}%")
        if losses:
            print(f"  Avg loss: {statistics.mean([t['pnl_pct'] for t in losses]):.2f}%")
        
        total_wins = sum(t['pnl_dollar'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl_dollar'] for t in losses)) if losses else 0.01
        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        print(f"  Profit Factor: {pf:.2f}")
        
        if version == 'v12':
            print(f"  🛡️ Blocked entries (events/VIX): {blocked_entries}")
            print(f"  📉 VIX-reduced entries: {vix_reduced}")
            # Macro multiplier analysis
            macro_trades = [t for t in trades if t.get('macro_mult', 1.0) < 1.0]
            full_trades = [t for t in trades if t.get('macro_mult', 1.0) >= 1.0]
            if macro_trades:
                m_win = sum(1 for t in macro_trades if t['pnl_pct'] > 0) / len(macro_trades) * 100
                print(f"  Macro-reduced trades: {len(macro_trades)} (Win%: {m_win:.0f}%)")
            if full_trades:
                f_win = sum(1 for t in full_trades if t['pnl_pct'] > 0) / len(full_trades) * 100
                print(f"  Full-size trades: {len(full_trades)} (Win%: {f_win:.0f}%)")
        
        # Exit breakdown
        reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            reasons[t['exit_reason']]['count'] += 1
            reasons[t['exit_reason']]['pnl'] += t['pnl_dollar']
        
        print(f"\n  Exit breakdown:")
        for r in ['TP', 'TRAIL', 'MAX_HOLD', 'STOP', 'OPEN']:
            if r in reasons:
                v = reasons[r]
                print(f"    {r:10s}: {v['count']:3d} trades, P&L: ${v['pnl']:+.2f}")
        
        # Vol tier analysis
        print(f"\n  Volatility tier:")
        for tier in ['high', 'mid', 'low']:
            tier_trades = [t for t in trades if t.get('vol_tier') == tier]
            if tier_trades:
                tier_pnl = sum(t['pnl_dollar'] for t in tier_trades)
                tier_wins = sum(1 for t in tier_trades if t['pnl_pct'] > 0)
                print(f"    {tier:5s}: {len(tier_trades)} trades, ${tier_pnl:+.2f}, {tier_wins}/{len(tier_trades)} wins")
        
        # Top/bottom tickers
        sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: -x[1])
        print(f"\n  Top 3:")
        for ticker, pnl in sorted_tickers[:3]:
            print(f"    🟢 {ticker}: ${pnl:+.2f}")
        print(f"  Bottom 3:")
        for ticker, pnl in sorted_tickers[-3:]:
            print(f"    {'🔴' if pnl < 0 else '🟡'} {ticker}: ${pnl:+.2f}")
        
        combined_results[strat_name] = {
            'return': total_return, 'trades': len(trades), 'win_pct': len(wins)/max(len(trades),1)*100,
            'pf': pf, 'max_dd': max_drawdown, 'final': final_value, 'capital': capital,
        }
    
    # Combined portfolio
    total_capital = sum(r['capital'] for r in combined_results.values())
    total_final = sum(r['final'] for r in combined_results.values())
    total_return = (total_final - total_capital) / total_capital * 100
    total_trades = sum(r['trades'] for r in combined_results.values())
    avg_win = statistics.mean([r['win_pct'] for r in combined_results.values()])
    avg_pf = statistics.mean([r['pf'] for r in combined_results.values()])
    worst_dd = max(r['max_dd'] for r in combined_results.values())
    
    print(f"\n{'═'*50}")
    print(f"  COMBINED {version}: ${total_capital:.0f} → ${total_final:.0f} ({total_return:+.1f}%)")
    print(f"  Total trades: {total_trades} | Avg Win%: {avg_win:.1f}% | Avg PF: {avg_pf:.2f}")
    print(f"  Worst strategy DD: {worst_dd:.1f}%")
    print(f"{'═'*50}")
    
    return {'return': total_return, 'trades': total_trades, 'win_pct': avg_win, 'pf': avg_pf, 'max_dd': worst_dd}


def main():
    print("█"*70)
    print("  PROJECT ARES v11 vs v12 (MACRO OVERLAY) BACKTEST")
    print("█"*70)
    
    # Fetch macro data
    print("\n📊 Fetching macro data...")
    vix_map = fetch_vix_data()
    print(f"  VIX: {len(vix_map)} days")
    spy_map = fetch_spy_data()
    print(f"  SPY: {len(spy_map)} days")
    
    # Fetch stock data
    print("\n📈 Fetching stock data from Supabase...")
    all_data = {}
    # Only fetch last 3 years of data (750 trading days) to save memory
    cutoff = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    for ticker in TICKERS:
        data = fetch('daily_prices', f'ticker=eq.{ticker}&date=gte.{cutoff}&order=date.asc')
        if len(data) < 250:
            print(f"  ⚠️ {ticker}: only {len(data)} days, skipping")
            continue
        all_data[ticker] = {
            'closes': [float(d['close']) for d in data],
            'highs': [float(d['high']) for d in data],
            'lows': [float(d['low']) for d in data],
            'volumes': [int(d['volume']) for d in data],
            'dates': [d['date'] for d in data],
        }
        print(f"  ✅ {ticker}: {len(data)} days")
    
    sample_ticker = list(all_data.keys())[0]
    total_days = len(all_data[sample_ticker]['dates'])
    
    # Test on validation period (last ~2 years = ~500 trading days)
    backtest_start = max(200, total_days - 500)
    
    # Run both versions
    v11_results = run_backtest('v11', vix_map, spy_map, all_data, backtest_start, total_days)
    v12_results = run_backtest('v12', vix_map, spy_map, all_data, backtest_start, total_days)
    
    # Comparison
    print(f"\n{'█'*70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'█'*70}")
    print(f"  {'Metric':<20s} {'v11':>12s} {'v12':>12s} {'Delta':>12s}")
    print(f"  {'─'*56}")
    
    for metric, label, fmt in [
        ('return', 'Return', '+.1f'),
        ('trades', 'Trades', '.0f'),
        ('win_pct', 'Win %', '.1f'),
        ('pf', 'Profit Factor', '.2f'),
        ('max_dd', 'Max Drawdown', '.1f'),
    ]:
        v11 = v11_results[metric]
        v12 = v12_results[metric]
        delta = v12 - v11
        sign = '+' if delta > 0 else ''
        # For drawdown, lower is better
        if metric == 'max_dd':
            better = '✅' if v12 < v11 else '❌'
        elif metric == 'trades':
            better = ''
        else:
            better = '✅' if v12 > v11 else '❌'
        print(f"  {label:<20s} {format(v11, fmt):>12s} {format(v12, fmt):>12s} {sign}{format(delta, fmt):>10s} {better}")
    
    print(f"\n{'█'*70}")

if __name__ == '__main__':
    main()
