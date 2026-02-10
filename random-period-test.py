#!/usr/bin/env python3
"""Test v11 on our 15 stocks across multiple random time periods"""

import json, statistics, urllib.request
from collections import defaultdict
from datetime import datetime

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

def fetch_yahoo(ticker, years=6):
    end = int(datetime.now().timestamp())
    start = end - (years * 365 * 86400)
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        result = data['chart']['result'][0]
        ts = result['timestamp']
        q = result['indicators']['quote'][0]
        rows = []
        for i in range(len(ts)):
            if q['close'][i] is None: continue
            rows.append({
                'date': datetime.fromtimestamp(ts[i]).strftime('%Y-%m-%d'),
                'close': q['close'][i], 'high': q['high'][i] or q['close'][i],
                'low': q['low'][i] or q['close'][i], 'volume': q['volume'][i] or 0,
            })
        return rows
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return []

def calc_rsi(closes, idx, period=14):
    if idx < period + 1: return 50
    g, l = 0, 0
    for i in range(idx - period, idx):
        d = closes[i+1] - closes[i]
        if d > 0: g += d
        else: l -= d
    ag, al = g/period, l/period
    if al == 0: return 100
    return 100 - 100/(1+ag/al)

def calc_atr(h, l, c, idx, p=14):
    trs = [max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(idx-p, idx)]
    return sum(trs)/len(trs) if trs else 1

def get_vol_tier(atr_pct):
    if atr_pct > 2.5: return 'high'
    elif atr_pct > 1.5: return 'mid'
    return 'low'

def get_adaptive(vt, s):
    s = dict(s)
    if vt == 'high':
        s['stop_mult'] = max(s['stop_mult'], 2.5); s['trail_dist'] = max(s['trail_dist'], 3.0); s['trail_start'] = max(s['trail_start'], 3.0)
    elif vt == 'low':
        s['stop_mult'] = min(s['stop_mult'], 1.2); s['tp_pct'] = min(s['tp_pct'], 0.12); s['trail_start'] = min(s['trail_start'], 1.5); s['trail_dist'] = min(s['trail_dist'], 1.5)
    return s

def check_vbounce(closes, idx):
    lb = min(250, idx-15)
    if lb < 50: return None
    b, d = 0, 0
    i = idx - lb
    while i < idx - 10:
        if calc_rsi(closes, i) < 35:
            d += 1
            mx = max(closes[i+1:i+11])
            if (mx - closes[i])/closes[i] > 0.05: b += 1
            i += 10
        else: i += 1
    if d < 3: return None
    return b/d >= 0.6

def check_regime(spy_closes, idx):
    """Bear market kill switch: SPY below declining SMA200 = no trading"""
    if idx < 220 or len(spy_closes) <= idx: return True  # default: allow
    sma200 = sum(spy_closes[idx-200:idx])/200
    sma200_prev = sum(spy_closes[idx-220:idx-20])/200  # SMA200 from 20 days ago
    spy_below = spy_closes[idx] < sma200
    sma_declining = sma200 < sma200_prev
    if spy_below and sma_declining:
        return False  # BEAR — don't trade
    return True  # OK to trade

def confluence(c, h, l, v, idx, td):
    if idx < 200: return 0, 50, 1, 2
    price = c[idx]
    sma50 = sum(c[idx-50:idx])/50
    sma200 = sum(c[idx-200:idx])/200
    rsi = calc_rsi(c, idx)
    roc5 = (c[idx]-c[idx-5])/c[idx-5]*100
    avg_v = sum(v[idx-20:idx])/20
    vr = v[idx]/avg_v if avg_v > 0 else 1
    atr = calc_atr(h, l, c, idx)
    atr_pct = atr/price*100
    
    votes = 0
    if rsi <= 35: votes += 1
    d50 = (price-sma50)/sma50*100
    if -8 < d50 < 0: votes += 1
    if price > sma200: votes += 1
    if -3 < roc5 < 3: votes += 1
    if vr > 1.3 and rsi < 45: votes += 1
    if td.get('peg', 1.5) < 1.5: votes += 1
    if td.get('beat_rate', 0.75) >= 0.75: votes += 1
    l52 = min(c[max(0,idx-252):idx+1]); h52 = max(c[max(0,idx-252):idx+1])
    if h52 > l52 and 0.2 <= (price-l52)/(h52-l52) <= 0.6: votes += 1
    if atr_pct < 4: votes += 1
    if check_vbounce(c, idx): votes += 1
    return votes, rsi, atr, atr_pct

def run_period(all_data, start_date, end_date, label, spy_closes=None, use_regime=False):
    print(f"\n{'─'*55}")
    print(f"  {label}: {start_date} → {end_date}")
    print(f"{'─'*55}")
    
    # Trim data to period (but keep 200 days before for SMA200)
    period_data = {}
    for tk in all_data:
        dates = all_data[tk]['d']
        # Find end index
        end_idx = next((i for i, d in enumerate(dates) if d > end_date), len(dates))
        # Keep all data up to end (need history for indicators)
        period_data[tk] = {k: all_data[tk][k][:end_idx] for k in ['c','h','l','v','d']}
    
    # Find start index
    sample = list(period_data.values())[0]
    start_idx = next((i for i, d in enumerate(sample['d']) if d >= start_date), 200)
    start_idx = max(200, start_idx)  # need 200 for SMA200
    total = len(sample['d'])
    
    actual_start = sample['d'][start_idx]
    actual_end = sample['d'][-1]
    days = total - start_idx
    print(f"  Actual: {actual_start} → {actual_end} ({days} trading days)")
    
    combined_final = 0
    all_trades = []
    
    for sn, sc in STRATEGIES.items():
        capital = 10000 * sc['alloc']
        cash = capital
        positions = {}
        trades = []
        
        for di in range(start_idx, total):
            for tk in list(positions.keys()):
                d = period_data[tk]
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
                elif (price-pos['ep'])/pos['ep'] >= pos['tp']: er = 'TP'
                elif di - pos['ei'] >= pos['mh']: er = 'MAX_HOLD'
                if er:
                    pnl = (price-pos['ep'])*pos['sh']
                    cash += pos['sh']*price
                    trades.append({'t':tk,'pnl':pnl,'er':er,'pnl_p':(price-pos['ep'])/pos['ep']*100})
                    del positions[tk]
            
            for tk in TICKERS:
                if tk in positions or tk not in period_data: continue
                d = period_data[tk]
                if di >= len(d['c']): continue
                # REGIME FILTER
                if use_regime and spy_closes and not check_regime(spy_closes, di):
                    continue
                price = d['c'][di]
                votes, rsi, atr, atr_pct = confluence(d['c'], d['h'], d['l'], d['v'], di, TICKER_DATA.get(tk,{}))
                if votes < 6: continue
                if rsi > 50:
                    sma50 = sum(d['c'][di-50:di])/50
                    if price > sma50: continue
                sz = sc['sizing'].get(min(votes,9), sc['sizing'][6])
                pv = capital*sz
                if pv > cash or pv < 50: continue
                sh = int(pv/price)
                if sh < 1: continue
                vt = get_vol_tier(atr_pct)
                ap = get_adaptive(vt, sc)
                positions[tk] = {
                    'ep':price,'ei':di,'sh':sh,'sl':price-ap['stop_mult']*atr,
                    'ta':price+ap['trail_start']*atr,'td':ap['trail_dist']*atr,
                    'th':price,'ts':None,'tp':ap['tp_pct'],'mh':ap['hold']
                }
                cash -= sh*price
        
        for tk in list(positions.keys()):
            d = period_data[tk]
            pos = positions[tk]
            price = d['c'][-1]
            pnl = (price-pos['ep'])*pos['sh']
            cash += pos['sh']*price
            trades.append({'t':tk,'pnl':pnl,'er':'OPEN','pnl_p':(price-pos['ep'])/pos['ep']*100})
        
        final = cash
        combined_final += final
        all_trades.extend(trades)
    
    cret = (combined_final - 10000) / 10000 * 100
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    stops = sum(1 for t in all_trades if t['er'] == 'STOP')
    tw = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
    tl = abs(sum(t['pnl'] for t in all_trades if t['pnl'] <= 0)) or 0.01
    pf = tw/tl
    
    # Per ticker
    tpnl = defaultdict(float)
    for t in all_trades: tpnl[t['t']] += t['pnl']
    best3 = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted(tpnl.items(), key=lambda x:-x[1])[:3])
    worst3 = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted(tpnl.items(), key=lambda x:x[1])[:3])
    
    print(f"  $10k → ${combined_final:.0f} ({cret:+.1f}%) | {len(all_trades)}t | {wins} wins ({wins/max(len(all_trades),1)*100:.0f}%) | PF {pf:.2f} | {stops} stops")
    print(f"  Best:  {best3}")
    print(f"  Worst: {worst3}")
    
    return cret, len(all_trades), wins/max(len(all_trades),1)*100, pf

# MAIN
print("█"*60)
print("  v11 ON OUR 15 STOCKS — RANDOM PERIODS")
print("█"*60)

print("\nFetching 6 years of data...")
all_data = {}
for tk in TICKERS:
    rows = fetch_yahoo(tk, years=6)
    if len(rows) < 500:
        print(f"  ⚠️ {tk}: {len(rows)} days, skipping")
        continue
    all_data[tk] = {
        'c': [r['close'] for r in rows], 'h': [r['high'] for r in rows],
        'l': [r['low'] for r in rows], 'v': [r['volume'] for r in rows],
        'd': [r['date'] for r in rows],
    }
    print(f"  ✅ {tk}: {len(rows)} days ({rows[0]['date']} → {rows[-1]['date']})")

# Fetch SPY for regime filter
print("\nFetching SPY for regime filter...")
spy_rows = fetch_yahoo('SPY', years=6)
spy_closes = [r['close'] for r in spy_rows]
spy_dates = [r['date'] for r in spy_rows]
print(f"  ✅ SPY: {len(spy_rows)} days ({spy_dates[0]} → {spy_dates[-1]})")

periods = [
    ('2022-01-01', '2022-06-30', 'H1 2022 — BEAR (rate hikes, crypto crash)'),
    ('2022-07-01', '2022-12-31', 'H2 2022 — BEAR BOTTOM + early recovery'),
    ('2023-01-01', '2023-06-30', 'H1 2023 — AI RALLY (ChatGPT era)'),
    ('2023-07-01', '2023-12-31', 'H2 2023 — CONSOLIDATION + year-end rally'),
    ('2024-01-01', '2024-06-30', 'H1 2024 — BULL continuation'),
    ('2024-07-01', '2024-12-31', 'H2 2024 — ELECTION + volatility'),
    ('2025-01-01', '2025-06-30', 'H1 2025 — RECENT'),
    ('2025-07-01', '2026-02-10', 'H2 2025-now — LATEST'),
]

# Run WITHOUT regime filter (v11)
print(f"\n{'█'*60}")
print("  RUN 1: v11 WITHOUT regime filter")
print(f"{'█'*60}")
results_no = []
for start, end, label in periods:
    try:
        r = run_period(all_data, start, end, label, spy_closes=None, use_regime=False)
        results_no.append((label, *r))
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Run WITH regime filter (v11.1)
print(f"\n{'█'*60}")
print("  RUN 2: v11.1 WITH regime filter (SPY < declining SMA200 = no trade)")
print(f"{'█'*60}")
results_yes = []
for start, end, label in periods:
    try:
        r = run_period(all_data, start, end, label, spy_closes=spy_closes, use_regime=True)
        results_yes.append((label, *r))
    except Exception as e:
        print(f"  ❌ Error: {e}")

# COMPARISON
print(f"\n{'='*80}")
print("  COMPARISON: v11 vs v11.1 (with regime filter)")
print(f"{'='*80}")
print(f"  {'Period':<35s} {'v11':>10s} {'v11.1':>10s} {'Diff':>10s} {'v11 PF':>8s} {'v11.1 PF':>8s}")
print(f"  {'─'*85}")

total_no, total_yes = 0, 0
for i in range(len(results_no)):
    ln = results_no[i][0][:33]
    rn = results_no[i][1]
    ry = results_yes[i][1] if i < len(results_yes) else 0
    pn = results_no[i][4]
    py = results_yes[i][4] if i < len(results_yes) else 0
    diff = ry - rn
    marker = '🟢' if diff > 0 else '🔴' if diff < -1 else '🟡'
    print(f"  {marker} {ln:<33s} {rn:>+9.1f}% {ry:>+9.1f}% {diff:>+9.1f}% {pn:>7.2f} {py:>7.2f}")
    total_no += rn
    total_yes += ry

print(f"  {'─'*85}")
avg_no = total_no / len(results_no)
avg_yes = total_yes / len(results_yes) if results_yes else 0
print(f"  {'AVERAGE':<35s} {avg_no:>+9.1f}% {avg_yes:>+9.1f}% {avg_yes-avg_no:>+9.1f}%")
prof_no = sum(1 for _, r, *_ in results_no if r > 0)
prof_yes = sum(1 for _, r, *_ in results_yes if r > 0)
print(f"  Profitable periods: v11={prof_no}/8, v11.1={prof_yes}/8")
