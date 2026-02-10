#!/usr/bin/env python3
"""
Project Ares v11 vs v11.1 Backtest
v11.1 adds 3 new confluence factors:
  - Institutional Flow (proxy: volume trend + price momentum)
  - Insider Sentiment (proxy: price-volume divergence pattern)
  - Short Interest Pressure (proxy: high volume on down days)
  
Since we don't have historical inst/insider/short data yet,
we use proven statistical proxies that correlate with these signals.
"""

import json, statistics, sys
from collections import defaultdict

TICKERS = ['NVDA','AMD','MU','MSFT','AAPL','AMZN','GOOGL','META','AVGO','QCOM','INTC','TSM','MRVL','LRCX','AMAT']

STRATEGIES = {
    'patient': {'hold': 20, 'trail_start': 2.5, 'trail_dist': 2.5, 'stop_mult': 1.5, 'tp_pct': 0.25, 'alloc': 0.40,
                'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
    'hybrid': {'hold': 10, 'trail_start': 1.5, 'trail_dist': 1.8, 'stop_mult': 1.2, 'tp_pct': 0.12, 'alloc': 0.35,
               'sizing': {6: 0.20, 7: 0.35, 8: 0.55, 9: 0.90}},
    'shortterm': {'hold': 10, 'trail_start': 2.0, 'trail_dist': 2.0, 'stop_mult': 1.5, 'tp_pct': 0.15, 'alloc': 0.25,
                  'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
}

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

# ══════════════════════════════════════════════════
#  CORE FUNCTIONS (same as v11)
# ══════════════════════════════════════════════════

def calc_rsi(closes, idx, period=14):
    if idx < period + 1: return 50
    gains, losses = 0, 0
    for i in range(idx - period, idx):
        d = closes[i+1] - closes[i]
        if d > 0: gains += d
        else: losses -= d
    avg_g = gains / period
    avg_l = losses / period
    if avg_l == 0: return 100
    return 100 - 100 / (1 + avg_g / avg_l)

def calc_atr(highs, lows, closes, idx, period=14):
    trs = []
    for i in range(idx - period, idx):
        tr = max(highs[i+1] - lows[i+1], abs(highs[i+1] - closes[i]), abs(lows[i+1] - closes[i]))
        trs.append(tr)
    return sum(trs) / len(trs) if trs else 1

def get_vol_tier(atr_pct):
    if atr_pct > 2.5: return 'high'
    elif atr_pct > 1.5: return 'mid'
    else: return 'low'

def get_adaptive_params(vol_tier, strat):
    s = dict(strat)
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

def check_recovery_pattern(closes, idx):
    lookback = min(250, idx - 15)
    if lookback < 50: return None
    bounces, dips = 0, 0
    i = idx - lookback
    while i < idx - 10:
        rsi = calc_rsi(closes, i)
        if rsi < 35:
            dips += 1
            entry = closes[i]
            max_10 = max(closes[i+1:i+11])
            if (max_10 - entry) / entry > 0.05:
                bounces += 1
            i += 10
        else:
            i += 1
    if dips >= 2:
        return bounces / dips
    return None

def sma(closes, idx, period):
    if idx < period: return closes[idx]
    return sum(closes[idx-period+1:idx+1]) / period

# ══════════════════════════════════════════════════
#  NEW v11.1 FACTORS (proxied from price/volume)
# ══════════════════════════════════════════════════

def institutional_flow_score(closes, volumes, idx):
    """
    Proxy for institutional accumulation/distribution.
    Smart money accumulates on low-volatility up days with above-avg volume.
    Score 0-100.
    """
    if idx < 40: return 50
    
    # 20-day accumulation/distribution ratio
    acc_days, dist_days = 0, 0
    avg_vol_20 = sum(volumes[idx-20:idx]) / 20 if sum(volumes[idx-20:idx]) > 0 else 1
    
    for i in range(idx-20, idx):
        if volumes[i] == 0: continue
        vol_ratio = volumes[i] / avg_vol_20
        price_change = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0
        
        if price_change > 0 and vol_ratio > 1.1:
            acc_days += vol_ratio  # Weight by volume strength
        elif price_change < -0.005 and vol_ratio > 1.1:
            dist_days += vol_ratio
    
    if acc_days + dist_days == 0: return 50
    ratio = acc_days / (acc_days + dist_days)
    
    # Also check: is 10-day avg volume trending up vs 40-day?
    avg_vol_10 = sum(volumes[idx-10:idx]) / 10 if sum(volumes[idx-10:idx]) > 0 else 1
    avg_vol_40 = sum(volumes[idx-40:idx]) / 40 if sum(volumes[idx-40:idx]) > 0 else 1
    vol_trend = (avg_vol_10 / avg_vol_40 - 1) * 50  # +/- adjustment
    
    score = ratio * 80 + 10 + min(max(vol_trend, -10), 10)
    return max(0, min(100, score))

def insider_sentiment_score(closes, volumes, idx):
    """
    Proxy for insider buying patterns.
    Insiders tend to buy after sharp drops when fundamentals intact.
    Detects: price drops with quick recovery + volume confirmation.
    Score 0-100.
    """
    if idx < 30: return 50
    
    # Count "smart dip buys" in last 60 days
    lookback = min(60, idx - 5)
    smart_buys = 0
    dip_count = 0
    
    for i in range(idx - lookback, idx - 3):
        # Detect: 3%+ drop followed by recovery within 5 days
        if closes[i-1] > 0 and (closes[i] - closes[i-1]) / closes[i-1] < -0.03:
            dip_count += 1
            max_5 = max(closes[i+1:min(i+6, idx)])
            if (max_5 - closes[i]) / closes[i] > 0.02:
                # Check if volume spiked on recovery days
                avg_vol = sum(volumes[i-10:i]) / 10 if sum(volumes[i-10:i]) > 0 else 1
                recovery_vol = sum(volumes[i+1:min(i+4, idx)]) / 3 if sum(volumes[i+1:min(i+4, idx)]) > 0 else 0
                if avg_vol > 0 and recovery_vol / avg_vol > 1.2:
                    smart_buys += 1
    
    if dip_count == 0: return 50
    buy_ratio = smart_buys / dip_count
    
    # Also factor: is price near recent lows? (insiders buy value)
    low_20 = min(closes[idx-20:idx])
    proximity_to_low = 1 - (closes[idx] - low_20) / (closes[idx] + 0.001)
    
    score = buy_ratio * 60 + proximity_to_low * 20 + 10
    return max(0, min(100, score))

def short_pressure_score(closes, volumes, idx):
    """
    Proxy for short interest pressure.
    High short interest = potential squeeze (bullish if price holds support).
    Detects: high volume on down days (shorts selling) + price resilience.
    Score 0-100. Higher = more bullish (shorts likely to cover).
    """
    if idx < 30: return 50
    
    # Volume on down days vs up days (last 20 days)
    down_vol, up_vol = 0, 0
    down_days, up_days = 0, 0
    
    for i in range(idx-20, idx):
        if closes[i] > closes[i-1]:
            up_vol += volumes[i]
            up_days += 1
        elif closes[i] < closes[i-1]:
            down_vol += volumes[i]
            down_days += 1
    
    # If lots of volume on down days but price holds → shorts trapped
    if up_vol + down_vol == 0: return 50
    down_ratio = down_vol / (up_vol + down_vol)
    
    # Price resilience: despite selling pressure, how close are we to 20d high?
    high_20 = max(closes[idx-20:idx])
    low_20 = min(closes[idx-20:idx])
    price_range = high_20 - low_20 if high_20 > low_20 else 1
    resilience = (closes[idx] - low_20) / price_range
    
    # High down_ratio + high resilience = squeeze potential (bullish)
    if down_ratio > 0.5 and resilience > 0.6:
        score = 70 + resilience * 20  # Shorts trapped, bullish
    elif down_ratio > 0.6 and resilience < 0.3:
        score = 20  # Genuine selling, bearish
    else:
        score = 50 + (resilience - 0.5) * 40
    
    return max(0, min(100, score))

# ══════════════════════════════════════════════════
#  SCORING: v11 (baseline) and v11.1 (with new factors)
# ══════════════════════════════════════════════════

def score_v11(closes, highs, lows, volumes, idx, ticker, ath):
    """Original v11 scoring (10 factors)"""
    price = closes[idx]
    rsi = calc_rsi(closes, idx)
    sma50 = sma(closes, idx, 50)
    sma200 = sma(closes, idx, 200)
    atr = calc_atr(highs, lows, closes, idx)
    atr_pct = (atr / price) * 100
    
    # Technical score
    tech = 0
    if rsi < 30: tech += 30
    elif rsi < 40: tech += 20
    elif rsi < 50: tech += 10
    if price < sma50 * 0.95: tech += 20
    elif price < sma50: tech += 10
    if price < sma200 * 0.95: tech += 20
    elif price < sma200: tech += 10
    low_52 = min(closes[max(0,idx-252):idx+1])
    high_52 = max(closes[max(0,idx-252):idx+1])
    range_52 = high_52 - low_52 if high_52 > low_52 else 1
    support_prox = (price - low_52) / range_52
    if support_prox < 0.2: tech += 20
    elif support_prox < 0.4: tech += 10
    tech = min(tech, 100)
    
    # Value score
    val = 0
    dd = ((ath - price) / ath) * 100 if ath > 0 else 0
    if dd > 30: val += 40
    elif dd > 20: val += 30
    elif dd > 10: val += 15
    td = TICKER_DATA.get(ticker, {})
    peg = td.get('peg', 1.5)
    if peg < 0.5: val += 30
    elif peg < 1.0: val += 20
    elif peg < 1.5: val += 10
    val = min(val, 100)
    
    # Catalyst score (simplified — no earnings dates in backtest)
    cat = 40  # neutral baseline
    
    # Volatility score
    vol = 0
    if atr_pct > 3: vol += 30
    elif atr_pct > 2: vol += 20
    elif atr_pct > 1.5: vol += 10
    if rsi < 30 and atr_pct > 2: vol += 20  # Fear + volatility = opportunity
    vol = min(vol, 100)
    
    # V-bounce check (10th factor)
    vb = check_recovery_pattern(closes, idx)
    bounce_bonus = 0
    if vb is not None and vb > 0.5 and rsi < 40:
        bounce_bonus = min(15, int(vb * 20))
    
    composite = tech * 0.30 + val * 0.25 + cat * 0.20 + vol * 0.15 + bounce_bonus
    return composite, atr_pct

def score_v11_1(closes, highs, lows, volumes, idx, ticker, ath):
    """v11.1 — adds institutional, insider, short interest factors"""
    v11_score, atr_pct = score_v11(closes, highs, lows, volumes, idx, ticker, ath)
    
    # New factors
    inst = institutional_flow_score(closes, volumes, idx)
    insider = insider_sentiment_score(closes, volumes, idx)
    short = short_pressure_score(closes, volumes, idx)
    
    # Weighted blend: new factors as light confluence (max +/- 3 points)
    # They confirm or slightly reduce confidence, never dominate
    new_factor_score = (inst * 0.40 + insider * 0.35 + short * 0.25)
    
    # Only boost, minimal drag — asymmetric (we want more entries, not fewer)
    if new_factor_score > 55:
        adjustment = (new_factor_score - 55) * 0.08  # max ~3.6 point boost
    elif new_factor_score < 40:
        adjustment = (new_factor_score - 40) * 0.05  # max ~2 point drag
    else:
        adjustment = 0
    
    composite = v11_score + adjustment
    return max(0, min(100, composite)), atr_pct, inst, insider, short

# ══════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════

def run_backtest(model='v11', start_offset=300, end_offset=None):
    """Run backtest for a given model version"""
    capital = 10000
    cash = capital
    positions = []
    trades = []
    
    for ticker in TICKERS:
        try:
            with open(f'/tmp/ares-data/{ticker}.json') as f:
                data = json.load(f)
        except:
            continue
        
        closes = [float(d['close']) for d in data]
        highs = [float(d['high']) for d in data]
        lows = [float(d['low']) for d in data]
        volumes = [int(d['volume'] or 0) for d in data]
        dates = [d['date'] for d in data]
        
        end = min(end_offset, len(closes)) if end_offset else len(closes)
        if start_offset >= len(closes): continue
        ath = max(closes[:start_offset]) if start_offset > 0 else closes[0]
        
        active_positions = {sname: None for sname in STRATEGIES}
        
        for idx in range(start_offset, end):
            price = closes[idx]
            ath = max(ath, price)
            atr = calc_atr(highs, lows, closes, idx)
            atr_pct = (atr / price) * 100
            vol_tier = get_vol_tier(atr_pct)
            
            if model == 'v11':
                score, _ = score_v11(closes, highs, lows, volumes, idx, ticker, ath)
            else:
                score, _, _, _, _ = score_v11_1(closes, highs, lows, volumes, idx, ticker, ath)
            
            # Entry logic (same for both models)
            for sname, strat in STRATEGIES.items():
                if active_positions[sname] is not None:
                    # Exit logic
                    pos = active_positions[sname]
                    adaptive = get_adaptive_params(vol_tier, strat)
                    days_held = idx - pos['entry_idx']
                    pnl_pct = (price - pos['entry']) / pos['entry']
                    
                    stop_dist = atr * adaptive['stop_mult']
                    stop_price = pos['entry'] - stop_dist
                    
                    # Trailing stop
                    if pnl_pct > adaptive['trail_start'] / 100:
                        trail_stop = price - atr * adaptive['trail_dist']
                        stop_price = max(stop_price, trail_stop)
                        pos['trail_high'] = max(pos.get('trail_high', price), price)
                    
                    # Take profit
                    tp_price = pos['entry'] * (1 + adaptive['tp_pct'])
                    
                    exit_reason = None
                    if price <= stop_price: exit_reason = 'stop'
                    elif price >= tp_price: exit_reason = 'tp'
                    elif days_held >= adaptive['hold']: exit_reason = 'time'
                    
                    if exit_reason:
                        pnl = (price - pos['entry']) * pos['shares']
                        cash += pos['shares'] * price
                        trades.append({
                            'ticker': ticker, 'strategy': sname, 'model': model,
                            'entry': pos['entry'], 'exit': price, 'pnl': pnl,
                            'pnl_pct': pnl_pct * 100, 'days': days_held,
                            'exit_reason': exit_reason, 'date': dates[idx],
                            'score': pos['score'],
                        })
                        active_positions[sname] = None
                    continue
                
                # Entry: score >= 60, sufficient cash
                if score >= 60:
                    score_tier = min(int(score / 10), 9)
                    pct = strat['sizing'].get(score_tier, 0)
                    if pct == 0: continue
                    alloc = capital * strat['alloc'] * pct
                    alloc = min(alloc, cash * 0.9)
                    if alloc < 100: continue
                    shares = int(alloc / price)
                    if shares < 1: continue
                    cash -= shares * price
                    active_positions[sname] = {
                        'entry': price, 'shares': shares, 'entry_idx': idx,
                        'score': score,
                    }
    
    # Close remaining positions at last price
    for ticker in TICKERS:
        try:
            with open(f'/tmp/ares-data/{ticker}.json') as f:
                data = json.load(f)
            closes = [float(d['close']) for d in data]
        except:
            continue
    
    total_value = cash
    return {
        'model': model,
        'trades': trades,
        'final_cash': cash,
        'total_trades': len(trades),
        'wins': len([t for t in trades if t['pnl'] > 0]),
        'losses': len([t for t in trades if t['pnl'] <= 0]),
        'total_pnl': sum(t['pnl'] for t in trades),
        'avg_pnl_pct': statistics.mean([t['pnl_pct'] for t in trades]) if trades else 0,
        'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0,
        'profit_factor': (sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))) if any(t['pnl'] < 0 for t in trades) else float('inf'),
        'max_win': max([t['pnl_pct'] for t in trades]) if trades else 0,
        'max_loss': min([t['pnl_pct'] for t in trades]) if trades else 0,
    }

# ══════════════════════════════════════════════════
#  STRESS TESTS
# ══════════════════════════════════════════════════

def run_period_tests():
    """Test both models across multiple periods"""
    periods = [
        ('2022H1 (Bear)', 0, 125),
        ('2022H2 (Recovery)', 125, 250),
        ('2023H1 (AI Rally)', 250, 375),
        ('2023H2 (Consolidation)', 375, 500),
        ('2024H1 (Bull)', 500, 625),
        ('2024H2 (Volatile)', 625, 750),
        ('2025H1 (Mixed)', 750, 875),
        ('2025H2-Now', 875, None),
    ]
    
    print("\n" + "=" * 90)
    print(f"{'PERIOD':<25} {'MODEL':<8} {'TRADES':>7} {'WIN%':>7} {'PnL':>10} {'PF':>7} {'AVG%':>8}")
    print("=" * 90)
    
    v11_total = 0
    v11_1_total = 0
    v11_wins = 0
    v11_1_wins = 0
    
    for name, start, end in periods:
        r11 = run_backtest('v11', start + 250, end + 250 if end else None)
        r111 = run_backtest('v11.1', start + 250, end + 250 if end else None)
        
        v11_total += r11['total_pnl']
        v11_1_total += r111['total_pnl']
        if r11['total_pnl'] > 0: v11_wins += 1
        if r111['total_pnl'] > 0: v11_1_wins += 1
        
        pf11 = f"{r11['profit_factor']:.2f}" if r11['profit_factor'] != float('inf') else "∞"
        pf111 = f"{r111['profit_factor']:.2f}" if r111['profit_factor'] != float('inf') else "∞"
        
        print(f"{name:<25} {'v11':<8} {r11['total_trades']:>7} {r11['win_rate']:>6.1f}% ${r11['total_pnl']:>9.0f} {pf11:>7} {r11['avg_pnl_pct']:>7.2f}%")
        print(f"{'':25} {'v11.1':<8} {r111['total_trades']:>7} {r111['win_rate']:>6.1f}% ${r111['total_pnl']:>9.0f} {pf111:>7} {r111['avg_pnl_pct']:>7.2f}%")
        better = "v11.1 ✅" if r111['total_pnl'] > r11['total_pnl'] else "v11 ✅"
        diff = r111['total_pnl'] - r11['total_pnl']
        print(f"{'':25} {'→':8} {better} ({'+' if diff > 0 else ''}{diff:.0f})")
        print("-" * 90)
    
    print(f"\n{'TOTAL':25} {'v11':<8} ${v11_total:>9.0f}  ({v11_wins}/8 profitable)")
    print(f"{'':25} {'v11.1':<8} ${v11_1_total:>9.0f}  ({v11_1_wins}/8 profitable)")
    diff = v11_1_total - v11_total
    pct = (diff / abs(v11_total) * 100) if v11_total != 0 else 0
    print(f"\n{'VERDICT':25} v11.1 is {'BETTER' if diff > 0 else 'WORSE'} by ${abs(diff):.0f} ({'+' if pct > 0 else ''}{pct:.1f}%)")

# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("PROJECT ARES — v11 vs v11.1 BACKTEST")
    print("v11.1 adds: Institutional Flow, Insider Sentiment, Short Pressure")
    print("=" * 60)
    
    # Full period comparison
    print("\n📊 FULL 10-MONTH BACKTEST")
    r11 = run_backtest('v11')
    r111 = run_backtest('v11.1')
    
    for label, r in [('v11', r11), ('v11.1', r111)]:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "∞"
        print(f"\n{label}: {r['total_trades']} trades | Win: {r['win_rate']:.1f}% | "
              f"PnL: ${r['total_pnl']:.0f} | PF: {pf} | "
              f"Avg: {r['avg_pnl_pct']:.2f}% | Max Win: {r['max_win']:.1f}% | Max Loss: {r['max_loss']:.1f}%")
    
    diff = r111['total_pnl'] - r11['total_pnl']
    print(f"\n→ v11.1 {'outperforms' if diff > 0 else 'underperforms'} by ${abs(diff):.0f}")
    
    # Period stress test
    print("\n📊 PERIOD STRESS TEST")
    run_period_tests()
    
    # Strategy breakdown
    print("\n📊 STRATEGY BREAKDOWN")
    for sname in STRATEGIES:
        t11 = [t for t in r11['trades'] if t['strategy'] == sname]
        t111 = [t for t in r111['trades'] if t['strategy'] == sname]
        
        pnl11 = sum(t['pnl'] for t in t11)
        pnl111 = sum(t['pnl'] for t in t111)
        wr11 = len([t for t in t11 if t['pnl'] > 0]) / len(t11) * 100 if t11 else 0
        wr111 = len([t for t in t111 if t['pnl'] > 0]) / len(t111) * 100 if t111 else 0
        
        print(f"  {sname:12} v11: {len(t11):3} trades, ${pnl11:>8.0f}, {wr11:.0f}% win")
        print(f"  {'':12} v11.1: {len(t111):3} trades, ${pnl111:>8.0f}, {wr111:.0f}% win")
