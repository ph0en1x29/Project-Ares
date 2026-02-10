#!/usr/bin/env python3
"""
Project Ares v11 Backtest — Adaptive Stops + Recovery Filter
Run against last 6 months of data from Supabase
"""

import json, sys, statistics, urllib.request
from datetime import datetime, timedelta
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

def fetch(table, params=""):
    url = f"{SB_URL}/rest/v1/{table}?{params}"
    req = urllib.request.Request(url, headers={"apikey": SB_KEY, "Authorization": f"Bearer {SB_KEY}"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

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
    """Adaptive volatility tier — NEW in v11"""
    if atr_pct > 2.5:
        return 'high'    # NVDA, AMD, MU
    elif atr_pct > 1.5:
        return 'mid'     # AMZN, GOOGL, META, AVGO
    else:
        return 'low'     # AAPL, MSFT

def get_adaptive_params(vol_tier, base_strat):
    """Adjust stop/TP based on volatility tier — NEW in v11"""
    s = dict(base_strat)  # copy
    if vol_tier == 'high':
        s['stop_mult'] = max(s['stop_mult'], 2.5)     # wider stop
        s['trail_dist'] = max(s['trail_dist'], 3.0)    # wider trail
        s['trail_start'] = max(s['trail_start'], 3.0)  # later trail activation
    elif vol_tier == 'low':
        s['stop_mult'] = min(s['stop_mult'], 1.2)      # tighter stop
        s['tp_pct'] = min(s['tp_pct'], 0.12)            # lower TP (these stocks don't spike 25%)
        s['trail_start'] = min(s['trail_start'], 1.5)   # earlier trail
        s['trail_dist'] = min(s['trail_dist'], 1.5)     # tighter trail
    return s

def check_recovery_pattern(closes, idx, lookback=250):
    """
    10th confluence factor: V-bounce detection — NEW in v11
    Checks if stock tends to V-bounce after RSI drops below 35
    Returns True if stock bounces >5% within 10 days at least 60% of the time
    """
    if idx < lookback:
        return None  # not enough data
    
    bounces = 0
    total_dips = 0
    
    # Find past instances where RSI < 35
    for i in range(max(15, idx - lookback), idx - 10):
        sub = closes[i-14:i+1]
        if len(sub) < 15:
            continue
        rsi = calc_rsi(sub)
        if rsi < 35:
            total_dips += 1
            # Check if price recovered >5% within 10 days
            entry = closes[i]
            max_next_10 = max(closes[i+1:i+11]) if i + 11 <= len(closes) else entry
            recovery_pct = (max_next_10 - entry) / entry * 100
            if recovery_pct > 5:
                bounces += 1
            # Skip ahead to avoid counting same dip multiple times
    
    if total_dips < 3:
        return None  # not enough data points
    
    bounce_rate = bounces / total_dips
    return bounce_rate >= 0.6

def calc_confluence(closes, highs, lows, volumes, idx, ticker_data):
    """Calculate confluence score with v11 additions"""
    if idx < 200:
        return 0, []
    
    price = closes[idx]
    sma50 = statistics.mean(closes[idx-50:idx])
    sma200 = statistics.mean(closes[idx-200:idx])
    sma20 = statistics.mean(closes[idx-20:idx])
    rsi = calc_rsi(closes[:idx+1])
    roc5 = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
    avg_vol = statistics.mean(volumes[idx-20:idx])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    atr = calc_atr(highs[:idx+1], lows[:idx+1], closes[:idx+1])
    atr_pct = atr / price * 100
    
    votes = 0
    factors = []
    
    # 1. RSI oversold
    if rsi <= 35:
        votes += 1; factors.append('RSI≤35')
    
    # 2. Near SMA50
    d50 = (price - sma50) / sma50 * 100
    if -8 < d50 < 0:
        votes += 1; factors.append('Near SMA50')
    
    # 3. Above SMA200
    if price > sma200:
        votes += 1; factors.append('Above SMA200')
    
    # 4. Stabilizing momentum
    if -3 < roc5 < 3:
        votes += 1; factors.append('Stabilizing')
    
    # 5. Volume capitulation
    if vol_ratio > 1.3 and rsi < 45:
        votes += 1; factors.append('Vol+Oversold')
    
    # 6. PEG < 1.5 (use stored data)
    if ticker_data.get('peg', 1.5) < 1.5:
        votes += 1; factors.append('PEG<1.5')
    
    # 7. Earnings beat ≥ 75% (use stored data)
    if ticker_data.get('beat_rate', 0.75) >= 0.75:
        votes += 1; factors.append('Beats≥75%')
    
    # 8. 52-week range sweet spot
    low_52 = min(closes[max(0, idx-252):idx+1])
    high_52 = max(closes[max(0, idx-252):idx+1])
    if high_52 > low_52:
        pos = (price - low_52) / (high_52 - low_52)
        if 0.2 <= pos <= 0.6:
            votes += 1; factors.append('52w Sweet')
    
    # 9. Low volatility entry
    if atr_pct < 4:
        votes += 1; factors.append('Low Vol')
    
    # 10. V-bounce pattern (NEW v11)
    is_bouncer = check_recovery_pattern(closes, idx)
    if is_bouncer:
        votes += 1; factors.append('V-Bounce ✓')
    
    return votes, factors, rsi, atr, atr_pct, sma50, sma200

# Known fundamentals (approximate, for confluence)
TICKER_DATA = {
    'NVDA': {'peg': 0.82, 'beat_rate': 0.90},
    'AMD': {'peg': 1.1, 'beat_rate': 0.80},
    'MU': {'peg': 0.3, 'beat_rate': 0.75},
    'MSFT': {'peg': 1.8, 'beat_rate': 0.85},
    'AAPL': {'peg': 2.0, 'beat_rate': 0.80},
    'AMZN': {'peg': 1.2, 'beat_rate': 0.80},
    'GOOGL': {'peg': 1.1, 'beat_rate': 0.80},
    'META': {'peg': 0.9, 'beat_rate': 0.85},
    'AVGO': {'peg': 1.0, 'beat_rate': 0.85},
    'QCOM': {'peg': 1.3, 'beat_rate': 0.75},
    'INTC': {'peg': 2.5, 'beat_rate': 0.50},
    'TSM': {'peg': 0.8, 'beat_rate': 0.80},
    'MRVL': {'peg': 1.2, 'beat_rate': 0.75},
    'LRCX': {'peg': 1.1, 'beat_rate': 0.80},
    'AMAT': {'peg': 0.9, 'beat_rate': 0.85},
}

def run_backtest(version='v11'):
    """Run backtest for last 6 months with both v10.1 and v11 rules"""
    
    print(f"\n{'='*70}")
    print(f"PROJECT ARES {version} BACKTEST — Last 6 Months")
    print(f"{'='*70}")
    
    # Fetch all data (need 200+ days before the 6-month window for SMA200)
    all_data = {}
    for ticker in TICKERS:
        data = fetch('daily_prices', f'ticker=eq.{ticker}&order=date.asc')
        if len(data) < 250:
            print(f"⚠️  {ticker}: only {len(data)} days, need 250+. Skipping.")
            continue
        all_data[ticker] = {
            'closes': [float(d['close']) for d in data],
            'highs': [float(d['high']) for d in data],
            'lows': [float(d['low']) for d in data],
            'volumes': [int(d['volume']) for d in data],
            'dates': [d['date'] for d in data],
        }
        print(f"✅ {ticker}: {len(data)} days ({data[0]['date']} → {data[-1]['date']})")
    
    # Find the start index for "last 6 months" (~130 trading days from end)
    sample_ticker = list(all_data.keys())[0]
    total_days = len(all_data[sample_ticker]['dates'])
    backtest_start = max(200, total_days - 130)  # 6 months back, but need 200 for SMA200
    
    start_date = all_data[sample_ticker]['dates'][backtest_start]
    end_date = all_data[sample_ticker]['dates'][-1]
    print(f"\n📅 Backtest period: {start_date} → {end_date} ({total_days - backtest_start} trading days)")
    
    # Track results per strategy
    STARTING_CAPITAL = 10000
    
    for strat_name, strat_config in STRATEGIES.items():
        print(f"\n{'─'*50}")
        print(f"Strategy: {strat_config.get('name', strat_name).upper()} ({strat_config['alloc']*100:.0f}% allocation)")
        print(f"{'─'*50}")
        
        capital = STARTING_CAPITAL * strat_config['alloc']
        cash = capital
        positions = {}  # ticker -> position dict
        trades = []
        ticker_pnl = defaultdict(float)
        ticker_trades = defaultdict(list)
        
        for day_idx in range(backtest_start, total_days):
            # Check exits first
            for ticker in list(positions.keys()):
                if ticker not in all_data:
                    continue
                d = all_data[ticker]
                pos = positions[ticker]
                price = d['closes'][day_idx]
                
                # Update trail high
                if price > pos['trail_high']:
                    pos['trail_high'] = price
                
                # Activate / update trailing stop
                if pos['trail_high'] >= pos['trail_activation']:
                    new_trail = pos['trail_high'] - pos['trail_dist_abs']
                    if pos['trail_stop'] is None or new_trail > pos['trail_stop']:
                        pos['trail_stop'] = new_trail
                
                exit_reason = None
                exit_price = price
                
                # Stop loss
                if price <= pos['stop_loss']:
                    exit_reason = 'STOP'
                # Trailing stop
                elif pos['trail_stop'] and price <= pos['trail_stop']:
                    exit_reason = 'TRAIL'
                # Take profit
                elif (price - pos['entry_price']) / pos['entry_price'] >= pos['tp_pct']:
                    exit_reason = 'TP'
                # Max hold
                elif day_idx - pos['entry_idx'] >= pos['max_hold']:
                    exit_reason = 'MAX_HOLD'
                
                if exit_reason:
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                    pnl_dollar = (exit_price - pos['entry_price']) * pos['shares']
                    cash += pos['shares'] * exit_price
                    
                    trade = {
                        'ticker': ticker,
                        'entry_date': d['dates'][pos['entry_idx']],
                        'exit_date': d['dates'][day_idx],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'exit_reason': exit_reason,
                        'confluence': pos['confluence'],
                        'vol_tier': pos.get('vol_tier', '?'),
                        'days': day_idx - pos['entry_idx'],
                    }
                    trades.append(trade)
                    ticker_pnl[ticker] += pnl_dollar
                    ticker_trades[ticker].append(trade)
                    del positions[ticker]
            
            # Check entries
            for ticker in TICKERS:
                if ticker in positions or ticker not in all_data:
                    continue
                
                d = all_data[ticker]
                if day_idx >= len(d['closes']):
                    continue
                
                price = d['closes'][day_idx]
                
                result = calc_confluence(
                    d['closes'], d['highs'], d['lows'], d['volumes'],
                    day_idx, TICKER_DATA.get(ticker, {})
                )
                votes, factors, rsi, atr, atr_pct, sma50, sma200 = result
                
                min_votes = 6
                if version == 'v11':
                    # v11: 6/10 (with recovery pattern as 10th factor)
                    min_votes = 6
                
                if votes < min_votes:
                    continue
                
                # Composite score check (simplified)
                if rsi > 50 and price > sma50:
                    continue  # not really a pullback
                
                # Position sizing
                sizing = strat_config['sizing']
                size_pct = sizing.get(min(votes, 9), sizing[6])
                pos_value = capital * size_pct
                
                if pos_value > cash or pos_value < 50:
                    continue
                
                shares = int(pos_value / price)
                if shares < 1:
                    continue
                
                # v11: Adaptive stops based on volatility tier
                if version == 'v11':
                    vol_tier = get_vol_tier(atr_pct)
                    adapted = get_adaptive_params(vol_tier, strat_config)
                    stop_mult = adapted['stop_mult']
                    trail_start_mult = adapted['trail_start']
                    trail_dist_mult = adapted['trail_dist']
                    tp_pct = adapted['tp_pct']
                    max_hold = adapted['hold']
                else:
                    vol_tier = 'n/a'
                    stop_mult = strat_config['stop_mult']
                    trail_start_mult = strat_config['trail_start']
                    trail_dist_mult = strat_config['trail_dist']
                    tp_pct = strat_config['tp_pct']
                    max_hold = strat_config['hold']
                
                stop_loss = price - stop_mult * atr
                trail_activation = price + trail_start_mult * atr
                trail_dist_abs = trail_dist_mult * atr
                
                positions[ticker] = {
                    'entry_price': price,
                    'entry_idx': day_idx,
                    'shares': shares,
                    'stop_loss': stop_loss,
                    'trail_activation': trail_activation,
                    'trail_dist_abs': trail_dist_abs,
                    'trail_high': price,
                    'trail_stop': None,
                    'tp_pct': tp_pct,
                    'max_hold': max_hold,
                    'confluence': votes,
                    'vol_tier': vol_tier,
                }
                
                cash -= shares * price
        
        # Close any remaining positions at last price
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
            })
            ticker_pnl[ticker] += pnl_dollar
            ticker_trades[ticker].append(trades[-1])
        
        # Results
        final_value = cash
        total_return = (final_value - capital) / capital * 100
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        
        print(f"\n📊 Results:")
        print(f"  Starting: ${capital:.0f} → Final: ${final_value:.0f} ({total_return:+.1f}%)")
        print(f"  Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)} | Win%: {len(wins)/max(len(trades),1)*100:.1f}%")
        
        if wins:
            print(f"  Avg win: +{statistics.mean([t['pnl_pct'] for t in wins]):.2f}%")
        if losses:
            print(f"  Avg loss: {statistics.mean([t['pnl_pct'] for t in losses]):.2f}%")
        
        total_wins = sum(t['pnl_dollar'] for t in wins) if wins else 0
        total_losses = abs(sum(t['pnl_dollar'] for t in losses)) if losses else 0.01
        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        print(f"  Profit Factor: {pf:.2f}")
        
        # Exit reason breakdown
        reasons = defaultdict(lambda: {'count': 0, 'pnl': 0})
        for t in trades:
            reasons[t['exit_reason']]['count'] += 1
            reasons[t['exit_reason']]['pnl'] += t['pnl_dollar']
        
        print(f"\n  Exit breakdown:")
        for r in ['TP', 'TRAIL', 'MAX_HOLD', 'STOP', 'OPEN']:
            if r in reasons:
                v = reasons[r]
                print(f"    {r:10s}: {v['count']:3d} trades, P&L: ${v['pnl']:+.2f}")
        
        # Per-ticker P&L
        print(f"\n  Per-ticker P&L (top 5 / bottom 5):")
        sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: -x[1])
        for ticker, pnl in sorted_tickers[:5]:
            t_trades = ticker_trades[ticker]
            t_wins = sum(1 for t in t_trades if t['pnl_pct'] > 0)
            print(f"    🟢 {ticker:5s}: ${pnl:+8.2f} ({len(t_trades)} trades, {t_wins}/{len(t_trades)} wins)")
        print(f"    {'─'*40}")
        for ticker, pnl in sorted_tickers[-5:]:
            t_trades = ticker_trades[ticker]
            t_wins = sum(1 for t in t_trades if t['pnl_pct'] > 0)
            print(f"    {'🔴' if pnl < 0 else '🟡'} {ticker:5s}: ${pnl:+8.2f} ({len(t_trades)} trades, {t_wins}/{len(t_trades)} wins)")
        
        # v11 specific: vol tier analysis
        if version == 'v11':
            print(f"\n  Volatility tier analysis:")
            for tier in ['high', 'mid', 'low']:
                tier_trades = [t for t in trades if t.get('vol_tier') == tier]
                if tier_trades:
                    tier_pnl = sum(t['pnl_dollar'] for t in tier_trades)
                    tier_wins = sum(1 for t in tier_trades if t['pnl_pct'] > 0)
                    tier_stops = sum(1 for t in tier_trades if t['exit_reason'] == 'STOP')
                    print(f"    {tier:5s}: {len(tier_trades)} trades, ${tier_pnl:+.2f}, {tier_wins}/{len(tier_trades)} wins, {tier_stops} stops")

# Run both versions
print("\n" + "█"*70)
print("  RUNNING v10.1 (CURRENT) vs v11 (ADAPTIVE STOPS + RECOVERY FILTER)")
print("█"*70)

run_backtest('v10.1')
run_backtest('v11')

print(f"\n{'='*70}")
print("COMPARISON COMPLETE")
print(f"{'='*70}")
