#!/usr/bin/env python3
"""Project Ares v10.1 vs v11 Backtest — reads from local JSON files"""

import json, statistics
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
    """Check if stock V-bounces after RSI < 35"""
    lookback = min(250, idx - 15)
    if lookback < 50: return None
    
    bounces = 0
    dips = 0
    i = idx - lookback
    while i < idx - 10:
        rsi = calc_rsi(closes, i)
        if rsi < 35:
            dips += 1
            entry = closes[i]
            max_10 = max(closes[i+1:i+11])
            if (max_10 - entry) / entry > 0.05:
                bounces += 1
            i += 10  # skip ahead
        else:
            i += 1
    
    if dips < 3: return None
    return bounces / dips >= 0.6

def calc_confluence(closes, highs, lows, volumes, idx, td, version):
    if idx < 200: return 0, [], 50, 1, 2, 0, 0
    
    price = closes[idx]
    sma50 = sum(closes[idx-50:idx]) / 50
    sma200 = sum(closes[idx-200:idx]) / 200
    rsi = calc_rsi(closes, idx)
    roc5 = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100
    avg_vol = sum(volumes[idx-20:idx]) / 20
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1
    atr = calc_atr(highs, lows, closes, idx)
    atr_pct = atr / price * 100
    
    votes = 0
    factors = []
    
    if rsi <= 35: votes += 1; factors.append('RSI≤35')
    d50 = (price - sma50) / sma50 * 100
    if -8 < d50 < 0: votes += 1; factors.append('SMA50')
    if price > sma200: votes += 1; factors.append('SMA200')
    if -3 < roc5 < 3: votes += 1; factors.append('Stab')
    if vol_ratio > 1.3 and rsi < 45: votes += 1; factors.append('VolCap')
    if td.get('peg', 1.5) < 1.5: votes += 1; factors.append('PEG')
    if td.get('beat_rate', 0.75) >= 0.75: votes += 1; factors.append('Beats')
    
    low52 = min(closes[max(0,idx-252):idx+1])
    high52 = max(closes[max(0,idx-252):idx+1])
    if high52 > low52:
        pos = (price - low52) / (high52 - low52)
        if 0.2 <= pos <= 0.6: votes += 1; factors.append('52w')
    
    if atr_pct < 4: votes += 1; factors.append('LowVol')
    
    # v11: 10th factor
    if version == 'v11':
        bounce = check_recovery_pattern(closes, idx)
        if bounce: votes += 1; factors.append('V-Bounce')
    
    return votes, factors, rsi, atr, atr_pct, sma50, sma200

def run(version):
    print(f"\n{'='*60}")
    print(f"  PROJECT ARES {version} — 6-Month Backtest")
    print(f"{'='*60}")
    
    all_data = {}
    for t in TICKERS:
        try:
            with open(f'/tmp/ares-data/{t}.json') as f:
                data = json.load(f)
            if len(data) < 250:
                print(f"  ⚠️ {t}: {len(data)} days, skipping")
                continue
            all_data[t] = {
                'c': [float(d['close']) for d in data],
                'h': [float(d['high']) for d in data],
                'l': [float(d['low']) for d in data],
                'v': [int(d['volume']) for d in data],
                'd': [d['date'] for d in data],
            }
        except: continue
    
    sample = list(all_data.values())[0]
    total = len(sample['d'])
    bt_start = max(200, total - 210)  # ~10 months
    print(f"  Period: {sample['d'][bt_start]} → {sample['d'][-1]} ({total - bt_start} days)")
    print(f"  Tickers: {len(all_data)}")
    
    combined_trades = []
    combined_capital = 10000
    combined_final = 0
    
    for sn, sc in STRATEGIES.items():
        capital = 10000 * sc['alloc']
        cash = capital
        positions = {}
        trades = []
        ticker_pnl = defaultdict(float)
        
        for di in range(bt_start, total):
            # Exits
            for tk in list(positions.keys()):
                d = all_data[tk]
                if di >= len(d['c']): continue
                pos = positions[tk]
                price = d['c'][di]
                
                if price > pos['th']: pos['th'] = price
                if pos['th'] >= pos['ta']:
                    nt = pos['th'] - pos['td']
                    if pos['ts'] is None or nt > pos['ts']: pos['ts'] = nt
                
                er = None
                if price <= pos['sl']: er = 'STOP'
                elif pos['ts'] and price <= pos['ts']: er = 'TRAIL'
                elif (price - pos['ep']) / pos['ep'] >= pos['tp']: er = 'TP'
                elif di - pos['ei'] >= pos['mh']: er = 'MAX_HOLD'
                
                if er:
                    pnl = (price - pos['ep']) * pos['sh']
                    pnl_p = (price - pos['ep']) / pos['ep'] * 100
                    cash += pos['sh'] * price
                    trades.append({'t': tk, 'pnl': pnl, 'pnl_p': pnl_p, 'er': er,
                                   'vt': pos.get('vt','?'), 'conf': pos['conf'],
                                   'ed': d['d'][pos['ei']], 'xd': d['d'][di], 'days': di-pos['ei']})
                    ticker_pnl[tk] += pnl
                    del positions[tk]
            
            # Entries
            for tk in TICKERS:
                if tk in positions or tk not in all_data: continue
                d = all_data[tk]
                if di >= len(d['c']): continue
                
                price = d['c'][di]
                votes, factors, rsi, atr, atr_pct, sma50, sma200 = calc_confluence(
                    d['c'], d['h'], d['l'], d['v'], di, TICKER_DATA.get(tk, {}), version)
                
                if votes < 6: continue
                if rsi > 50 and price > sma50: continue
                
                sz = sc['sizing'].get(min(votes, 9), sc['sizing'][6])
                pv = capital * sz
                if pv > cash or pv < 50: continue
                sh = int(pv / price)
                if sh < 1: continue
                
                if version == 'v11':
                    vt = get_vol_tier(atr_pct)
                    ap = get_adaptive_params(vt, sc)
                    sm, tst, tdt, tp, mh = ap['stop_mult'], ap['trail_start'], ap['trail_dist'], ap['tp_pct'], ap['hold']
                else:
                    vt = 'n/a'
                    sm, tst, tdt, tp, mh = sc['stop_mult'], sc['trail_start'], sc['trail_dist'], sc['tp_pct'], sc['hold']
                
                positions[tk] = {
                    'ep': price, 'ei': di, 'sh': sh,
                    'sl': price - sm * atr, 'ta': price + tst * atr,
                    'td': tdt * atr, 'th': price, 'ts': None,
                    'tp': tp, 'mh': mh, 'conf': votes, 'vt': vt
                }
                cash -= sh * price
        
        # Close remaining
        for tk in list(positions.keys()):
            d = all_data[tk]
            pos = positions[tk]
            price = d['c'][-1]
            pnl = (price - pos['ep']) * pos['sh']
            cash += pos['sh'] * price
            trades.append({'t': tk, 'pnl': pnl, 'pnl_p': (price-pos['ep'])/pos['ep']*100,
                           'er': 'OPEN', 'vt': pos.get('vt','?'), 'conf': pos['conf'],
                           'ed': d['d'][pos['ei']], 'xd': d['d'][-1], 'days': total-1-pos['ei']})
            ticker_pnl[tk] += pnl
        
        final = cash
        ret = (final - capital) / capital * 100
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        tw = sum(t['pnl'] for t in wins) if wins else 0
        tl = abs(sum(t['pnl'] for t in losses)) if losses else 0.01
        pf = tw / tl
        
        combined_final += final
        combined_trades.extend(trades)
        
        print(f"\n  ── {sn.upper()} ({sc['alloc']*100:.0f}%) ──")
        print(f"  ${capital:.0f} → ${final:.0f} ({ret:+.1f}%) | {len(trades)} trades | Win: {len(wins)}/{len(trades)} ({len(wins)/max(len(trades),1)*100:.0f}%) | PF: {pf:.2f}")
        
        # Exit breakdown
        for er_name in ['TP','TRAIL','MAX_HOLD','STOP','OPEN']:
            er_trades = [t for t in trades if t['er'] == er_name]
            if er_trades:
                er_pnl = sum(t['pnl'] for t in er_trades)
                print(f"    {er_name:10s}: {len(er_trades):3d} trades, ${er_pnl:+.2f}")
        
        # Top/bottom tickers
        sorted_t = sorted(ticker_pnl.items(), key=lambda x: -x[1])
        top3 = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted_t[:3])
        bot3 = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted_t[-3:])
        print(f"    Best:  {top3}")
        print(f"    Worst: {bot3}")
        
        # v11 vol tier analysis
        if version == 'v11':
            for tier in ['high','mid','low']:
                tt = [t for t in trades if t.get('vt') == tier]
                if tt:
                    tp_sum = sum(t['pnl'] for t in tt)
                    tw_c = sum(1 for t in tt if t['pnl'] > 0)
                    stops = sum(1 for t in tt if t['er'] == 'STOP')
                    print(f"    Vol-{tier:4s}: {len(tt)} trades, ${tp_sum:+.2f}, {tw_c}/{len(tt)} wins, {stops} stops")
    
    # Combined summary
    cret = (combined_final - combined_capital) / combined_capital * 100
    cwins = sum(1 for t in combined_trades if t['pnl'] > 0)
    print(f"\n  ═══ COMBINED ═══")
    print(f"  $10,000 → ${combined_final:.0f} ({cret:+.1f}%)")
    print(f"  {len(combined_trades)} total trades | {cwins} wins ({cwins/max(len(combined_trades),1)*100:.0f}%)")
    
    # NVDA + AAPL specific
    for focus in ['NVDA', 'AAPL']:
        ft = [t for t in combined_trades if t['t'] == focus]
        if ft:
            fp = sum(t['pnl'] for t in ft)
            fw = sum(1 for t in ft if t['pnl'] > 0)
            fs = sum(1 for t in ft if t['er'] == 'STOP')
            print(f"  {focus}: ${fp:+.2f} | {len(ft)} trades | {fw} wins | {fs} stops")

run('v10.1')
run('v11')

print(f"\n{'='*60}")
print("  DONE — Compare v10.1 vs v11 above")
print(f"{'='*60}")
