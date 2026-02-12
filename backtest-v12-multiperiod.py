#!/usr/bin/env python3
"""
Project Ares v12 Multi-Period Backtest
Tests v11 vs v12 across: Super Short (3mo), Short (6mo), Medium (1yr), Long (2yr)
"""

import json, sys, statistics, urllib.request
from datetime import datetime, timedelta, date, timezone
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

PERIODS = {
    'Super Short (3mo)': 63,    # ~63 trading days
    'Short (6mo)': 126,
    'Medium (1yr)': 252,
    'Long (2yr)': 504,
}

# ─── MACRO DATA ──────────────────────────────────────────────

def fetch_vix_data():
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/%5EVIX?range=6y&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
    result = data['chart']['result'][0]
    vix_map = {}
    for t, c in zip(result['timestamp'], result['indicators']['quote'][0]['close']):
        if c is not None:
            d = datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d')
            vix_map[d] = round(c, 2)
    return vix_map

# Event dates — ONLY CPI + FOMC (v12.1: loosened from v12)
HIGH_IMPACT_EVENTS = set()

# CPI releases (approximate)
for year in range(2020, 2027):
    for month in range(1, 13):
        for day in [10, 11, 12, 13, 14, 15]:
            try:
                d = date(year, month, day)
                if d.weekday() < 5:
                    HIGH_IMPACT_EVENTS.add(d.isoformat())
                    prev = d - timedelta(days=1)
                    while prev.weekday() >= 5: prev -= timedelta(days=1)
                    HIGH_IMPACT_EVENTS.add(prev.isoformat())
                    break
            except ValueError:
                continue

FOMC_DATES = [
    '2020-01-29','2020-03-03','2020-03-15','2020-04-29','2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16','2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15','2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12','2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-11-05','2025-12-17',
    '2026-01-28','2026-03-18','2026-05-06','2026-06-17','2026-07-29','2026-09-16','2026-10-28','2026-12-09',
]
for fd in FOMC_DATES:
    HIGH_IMPACT_EVENTS.add(fd)
    d = date.fromisoformat(fd)
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5: prev -= timedelta(days=1)
    HIGH_IMPACT_EVENTS.add(prev.isoformat())

def is_event_blocked(date_str):
    return date_str in HIGH_IMPACT_EVENTS

def get_vix_regime(vix_val):
    if vix_val is None: return 'normal', 1.0
    if vix_val >= 30: return 'crisis', 0.0
    elif vix_val >= 25: return 'elevated', 0.5
    else: return 'normal', 1.0  # v12.1: removed 20-25 caution zone

def get_fg_multiplier(vix_val):
    """v12.1: Removed complacency penalty (VIX<15). Added contrarian boost."""
    if vix_val is None: return 1.0
    if vix_val < 20: return 1.0   # v12.1: no penalty for low VIX
    elif vix_val < 25: return 1.0
    elif vix_val < 30: return 1.0  # handled by regime
    return 1.0

def is_vix_declining(vix_map, date_str, lookback=5):
    d = date.fromisoformat(date_str)
    vals = []
    for i in range(lookback + 1):
        check = (d - timedelta(days=i)).isoformat()
        if check in vix_map: vals.append(vix_map[check])
    if len(vals) < 3: return False
    return vals[0] < vals[-1]

def get_contrarian_boost(vix_map, date_str, vix_val):
    """v12.1 NEW: If VIX declining from spike AND elevated, boost size"""
    if vix_val is None or vix_val < 22: return 1.0
    if is_vix_declining(vix_map, date_str):
        return 1.2  # 20% boost — mean reversion opportunity
    return 1.0

# ─── CORE ENGINE ──────────────────────────────────────────────

def fetch_sb(table, params=""):
    all_rows = []
    offset = 0
    while True:
        url = f"{SB_URL}/rest/v1/{table}?{params}&limit=1000&offset={offset}"
        req = urllib.request.Request(url, headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
        with urllib.request.urlopen(req) as r:
            rows = json.loads(r.read())
        all_rows.extend(rows)
        if len(rows) < 1000: break
        offset += 1000
    return all_rows

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-period-1+i] - closes[-period-1+i-1]
        gains.append(max(0, d)); losses.append(max(0, -d))
    avg_gain, avg_loss = statistics.mean(gains), statistics.mean(losses)
    if avg_loss == 0: return 100
    return 100 - 100 / (1 + avg_gain / avg_loss)

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
            max_next_10 = max(closes[i+1:i+11]) if i + 11 <= len(closes) else entry
            if (max_next_10 - entry) / entry * 100 > 5: bounces += 1
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

# ─── BACKTEST ENGINE ─────────────────────────────────────────

def run_single(version, vix_map, all_data, backtest_start, total_days):
    """Run backtest, return combined metrics"""
    sample_ticker = list(all_data.keys())[0]
    STARTING_CAPITAL = 10000
    all_trades = []
    total_final = 0
    worst_dd = 0

    for strat_name, strat_config in STRATEGIES.items():
        capital = STARTING_CAPITAL * strat_config['alloc']
        cash = capital
        positions = {}
        trades = []
        peak_value = capital
        max_drawdown = 0

        for day_idx in range(backtest_start, total_days):
            current_date = all_data[sample_ticker]['dates'][day_idx]
            vix_today = vix_map.get(current_date)

            # Portfolio value
            port_value = cash
            for ticker, pos in positions.items():
                if ticker in all_data and day_idx < len(all_data[ticker]['closes']):
                    port_value += pos['shares'] * all_data[ticker]['closes'][day_idx]
            if port_value > peak_value: peak_value = port_value
            dd = (peak_value - port_value) / peak_value * 100
            if dd > max_drawdown: max_drawdown = dd

            # Exits
            for ticker in list(positions.keys()):
                if ticker not in all_data: continue
                d = all_data[ticker]
                if day_idx >= len(d['closes']): continue
                pos = positions[ticker]
                price = d['closes'][day_idx]
                if price > pos['trail_high']: pos['trail_high'] = price
                if pos['trail_high'] >= pos['trail_activation']:
                    new_trail = pos['trail_high'] - pos['trail_dist_abs']
                    if pos['trail_stop'] is None or new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail

                exit_reason = None
                if price <= pos['stop_loss']: exit_reason = 'STOP'
                elif pos['trail_stop'] and price <= pos['trail_stop']: exit_reason = 'TRAIL'
                elif (price - pos['entry_price']) / pos['entry_price'] >= pos['tp_pct']: exit_reason = 'TP'
                elif day_idx - pos['entry_idx'] >= pos['max_hold']: exit_reason = 'MAX_HOLD'

                if exit_reason:
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
                    pnl_dollar = (price - pos['entry_price']) * pos['shares']
                    cash += pos['shares'] * price
                    trades.append({'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar, 'exit_reason': exit_reason})
                    del positions[ticker]

            # Entries
            for ticker in TICKERS:
                if ticker in positions or ticker not in all_data: continue
                d = all_data[ticker]
                if day_idx >= len(d['closes']): continue
                price = d['closes'][day_idx]
                result = calc_confluence(d['closes'], d['highs'], d['lows'], d['volumes'], day_idx, TICKER_DATA.get(ticker, {}))
                votes, factors, rsi, atr, atr_pct, sma50, sma200 = result
                if votes < 6: continue
                if rsi > 50 and price > sma50: continue

                macro_mult = 1.0
                if version == 'v12.1':
                    if is_event_blocked(current_date): continue
                    if vix_today is not None:
                        regime, regime_mult = get_vix_regime(vix_today)
                        if regime == 'crisis':
                            if not is_vix_declining(vix_map, current_date): continue
                            else: regime_mult = 0.5
                        macro_mult *= regime_mult
                        macro_mult *= get_contrarian_boost(vix_map, current_date, vix_today)

                sizing = strat_config['sizing']
                size_pct = sizing.get(min(votes, 9), sizing[6])
                pos_value = capital * size_pct * macro_mult
                if pos_value > cash or pos_value < 50: continue
                shares = int(pos_value / price)
                if shares < 1: continue

                vol_tier = get_vol_tier(atr_pct)
                adapted = get_adaptive_params(vol_tier, strat_config)
                stop_loss = price - adapted['stop_mult'] * atr
                trail_activation = price + adapted['trail_start'] * atr
                trail_dist_abs = adapted['trail_dist'] * atr

                positions[ticker] = {
                    'entry_price': price, 'entry_idx': day_idx, 'shares': shares,
                    'stop_loss': stop_loss, 'trail_activation': trail_activation,
                    'trail_dist_abs': trail_dist_abs, 'trail_high': price,
                    'trail_stop': None, 'tp_pct': adapted['tp_pct'], 'max_hold': adapted['hold'],
                    'confluence': votes, 'vol_tier': vol_tier,
                }
                cash -= shares * price

        # Close remaining
        for ticker in list(positions.keys()):
            d = all_data[ticker]
            pos = positions[ticker]
            price = d['closes'][-1]
            pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
            pnl_dollar = (price - pos['entry_price']) * pos['shares']
            cash += pos['shares'] * price
            trades.append({'pnl_pct': pnl_pct, 'pnl_dollar': pnl_dollar, 'exit_reason': 'OPEN'})

        total_final += cash
        if max_drawdown > worst_dd: worst_dd = max_drawdown
        all_trades.extend(trades)

    total_return = (total_final - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    wins = [t for t in all_trades if t['pnl_pct'] > 0]
    losses = [t for t in all_trades if t['pnl_pct'] <= 0]
    win_pct = len(wins) / max(len(all_trades), 1) * 100
    total_win_d = sum(t['pnl_dollar'] for t in wins) if wins else 0
    total_loss_d = abs(sum(t['pnl_dollar'] for t in losses)) if losses else 0.01
    pf = total_win_d / total_loss_d if total_loss_d > 0 else float('inf')
    avg_win = statistics.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = statistics.mean([t['pnl_pct'] for t in losses]) if losses else 0

    return {
        'return': total_return, 'trades': len(all_trades), 'win_pct': win_pct,
        'pf': pf, 'max_dd': worst_dd, 'avg_win': avg_win, 'avg_loss': avg_loss,
    }


def main():
    print("█" * 70)
    print("  ARES v11 vs v12.1 — MULTI-PERIOD BACKTEST")
    print("█" * 70)

    print("\n📊 Fetching macro data...")
    vix_map = fetch_vix_data()
    print(f"  VIX: {len(vix_map)} days")

    print("\n📈 Fetching stock data...")
    cutoff = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    all_data = {}
    for ticker in TICKERS:
        data = fetch_sb('daily_prices', f'ticker=eq.{ticker}&date=gte.{cutoff}&order=date.asc')
        if len(data) < 250:
            print(f"  ⚠️ {ticker}: {len(data)} days, skipping")
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

    print(f"\n{'█'*70}")
    print(f"  {'Period':<22s} │ {'Version':^7s} │ {'Return':>8s} │ {'Trades':>6s} │ {'Win%':>6s} │ {'PF':>6s} │ {'MaxDD':>6s} │ {'AvgW':>6s} │ {'AvgL':>6s}")
    print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")

    summary = {}
    for period_name, trading_days in PERIODS.items():
        backtest_start = max(200, total_days - trading_days)
        if backtest_start >= total_days - 20:
            print(f"  {period_name:<22s} │ SKIP — not enough data")
            continue

        actual_days = total_days - backtest_start
        start_date = all_data[sample_ticker]['dates'][backtest_start]
        end_date = all_data[sample_ticker]['dates'][-1]

        for version in ['v11', 'v12.1']:
            r = run_single(version, vix_map, all_data, backtest_start, total_days)
            label = f"{period_name}" if version == 'v11' else ""
            print(f"  {label:<22s} │ {version:^7s} │ {r['return']:>+7.1f}% │ {r['trades']:>6d} │ {r['win_pct']:>5.1f}% │ {r['pf']:>6.2f} │ {r['max_dd']:>5.1f}% │ {r['avg_win']:>+5.1f}% │ {r['avg_loss']:>+5.1f}%")
            summary[(period_name, version)] = r

        # Delta row
        v11 = summary[(period_name, 'v11')]
        v12 = summary[(period_name, 'v12.1')]
        ret_delta = v12['return'] - v11['return']
        dd_delta = v12['max_dd'] - v11['max_dd']
        pf_delta = v12['pf'] - v11['pf']
        ret_icon = '✅' if ret_delta > 0 else '❌'
        dd_icon = '✅' if dd_delta < 0 else '❌'
        pf_icon = '✅' if pf_delta > 0 else '❌'
        print(f"  {'  └ Δ (v12.1-v11)':<22s} │ {'delta':^7s} │ {ret_delta:>+7.1f}%{ret_icon}│ {v12['trades']-v11['trades']:>+6d} │ {v12['win_pct']-v11['win_pct']:>+5.1f}% │ {pf_delta:>+5.2f}{pf_icon}│ {dd_delta:>+5.1f}%{dd_icon}│        │")
        print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}")

    # Final verdict
    print(f"\n{'█'*70}")
    print("  VERDICT")
    print(f"{'█'*70}")
    
    wins_v12 = 0
    for period_name in PERIODS:
        if (period_name, 'v11') not in summary: continue
        v11 = summary[(period_name, 'v11')]
        v12 = summary[(period_name, 'v12.1')]
        
        better_return = '✅' if v12['return'] > v11['return'] else '❌'
        better_risk = '✅' if v12['max_dd'] < v11['max_dd'] else '❌'
        better_pf = '✅' if v12['pf'] > v11['pf'] else '❌'
        
        # v12.1 wins if it has better risk-adjusted return (PF + DD)
        risk_adj_better = v12['pf'] > v11['pf'] and v12['max_dd'] < v11['max_dd']
        if risk_adj_better: wins_v12 += 1
        
        print(f"  {period_name}: Return {better_return} | Risk {better_risk} | PF {better_pf} | Risk-Adj {'✅ v12.1' if risk_adj_better else '❌ v11'}")
    
    print(f"\n  v12.1 wins risk-adjusted in {wins_v12}/{len([p for p in PERIODS if (p, 'v11') in summary])} periods")
    print(f"{'█'*70}")


if __name__ == '__main__':
    main()
