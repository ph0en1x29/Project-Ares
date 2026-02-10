#!/usr/bin/env python3
"""
Stress Test — Run v11 on 10 UNSEEN stocks + 2022 bear market
Proves model isn't overfitted to our 15 watchlist stocks
"""

import json, statistics, urllib.request
from collections import defaultdict
from datetime import datetime

# 10 stocks we've NEVER tested the model on
# Mix of sectors: healthcare, finance, energy, consumer, industrial
UNSEEN = ['CRM','NFLX','COST','UNH','JPM','XOM','HD','LLY','CAT','PANW']

# Also test our original 15 but during 2022 BEAR MARKET
ORIGINAL = ['NVDA','AMD','MU','MSFT','AAPL','AMZN','GOOGL','META','AVGO','QCOM','INTC','TSM','MRVL','LRCX','AMAT']

def fetch_yahoo(ticker, years=4):
    """Fetch daily OHLCV from Yahoo Finance via CORS proxy"""
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
                'close': q['close'][i],
                'high': q['high'][i] or q['close'][i],
                'low': q['low'][i] or q['close'][i],
                'volume': q['volume'][i] or 0,
            })
        return rows
    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return []

def calc_rsi(closes, idx, period=14):
    if idx < period + 1: return 50
    gains, losses = 0, 0
    for i in range(idx - period, idx):
        d = closes[i+1] - closes[i]
        if d > 0: gains += d
        else: losses -= d
    ag = gains / period
    al = losses / period
    if al == 0: return 100
    return 100 - 100 / (1 + ag / al)

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
            if (max_10 - entry) / entry > 0.05: bounces += 1
            i += 10
        else: i += 1
    if dips < 3: return None
    return bounces / dips >= 0.6

def calc_confluence(closes, highs, lows, volumes, idx):
    if idx < 200: return 0, 50, 1, 2
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
    if rsi <= 35: votes += 1
    d50 = (price - sma50) / sma50 * 100
    if -8 < d50 < 0: votes += 1
    if price > sma200: votes += 1
    if -3 < roc5 < 3: votes += 1
    if vol_ratio > 1.3 and rsi < 45: votes += 1
    # Give 2 freebies for PEG + earnings (unknown for unseen stocks)
    votes += 2
    low52 = min(closes[max(0,idx-252):idx+1])
    high52 = max(closes[max(0,idx-252):idx+1])
    if high52 > low52:
        pos = (price - low52) / (high52 - low52)
        if 0.2 <= pos <= 0.6: votes += 1
    if atr_pct < 4: votes += 1
    bounce = check_recovery_pattern(closes, idx)
    if bounce: votes += 1
    
    return votes, rsi, atr, atr_pct

STRATEGIES = {
    'patient': {'hold': 20, 'trail_start': 2.5, 'trail_dist': 2.5, 'stop_mult': 1.5, 'tp_pct': 0.25, 'alloc': 0.40,
                'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
    'hybrid': {'hold': 10, 'trail_start': 1.5, 'trail_dist': 1.8, 'stop_mult': 1.2, 'tp_pct': 0.12, 'alloc': 0.35,
               'sizing': {6: 0.20, 7: 0.35, 8: 0.55, 9: 0.90}},
    'shortterm': {'hold': 10, 'trail_start': 2.0, 'trail_dist': 2.0, 'stop_mult': 1.5, 'tp_pct': 0.15, 'alloc': 0.25,
                  'sizing': {6: 0.15, 7: 0.30, 8: 0.50, 9: 0.90}},
}

def run_backtest(tickers_data, label, start_idx_offset=210):
    """Generic backtest runner"""
    # Find common length
    min_len = min(len(d['c']) for d in tickers_data.values())
    bt_start = max(200, min_len - start_idx_offset)
    
    sample = list(tickers_data.values())[0]
    print(f"  Period: {sample['d'][bt_start]} → {sample['d'][-1]} ({min_len - bt_start} days)")
    print(f"  Tickers: {list(tickers_data.keys())}")
    
    combined_capital = 10000
    combined_final = 0
    all_trades = []
    
    for sn, sc in STRATEGIES.items():
        capital = 10000 * sc['alloc']
        cash = capital
        positions = {}
        trades = []
        ticker_pnl = defaultdict(float)
        
        for di in range(bt_start, min_len):
            # Exits
            for tk in list(positions.keys()):
                d = tickers_data[tk]
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
                    cash += pos['sh'] * price
                    trades.append({'t': tk, 'pnl': pnl, 'pnl_p': (price-pos['ep'])/pos['ep']*100, 'er': er, 'vt': pos.get('vt','?')})
                    ticker_pnl[tk] += pnl
                    del positions[tk]
            
            # Entries
            for tk in tickers_data:
                if tk in positions: continue
                d = tickers_data[tk]
                if di >= len(d['c']): continue
                
                price = d['c'][di]
                votes, rsi, atr, atr_pct = calc_confluence(d['c'], d['h'], d['l'], d['v'], di)
                
                if votes < 6: continue
                if rsi > 50:
                    sma50 = sum(d['c'][di-50:di]) / 50
                    if price > sma50: continue
                
                sz = sc['sizing'].get(min(votes, 9), sc['sizing'][6])
                pv = capital * sz
                if pv > cash or pv < 50: continue
                sh = int(pv / price)
                if sh < 1: continue
                
                vt = get_vol_tier(atr_pct)
                ap = get_adaptive_params(vt, sc)
                
                positions[tk] = {
                    'ep': price, 'ei': di, 'sh': sh,
                    'sl': price - ap['stop_mult'] * atr, 'ta': price + ap['trail_start'] * atr,
                    'td': ap['trail_dist'] * atr, 'th': price, 'ts': None,
                    'tp': ap['tp_pct'], 'mh': ap['hold'], 'vt': vt
                }
                cash -= sh * price
        
        # Close remaining
        for tk in list(positions.keys()):
            d = tickers_data[tk]
            pos = positions[tk]
            price = d['c'][-1]
            pnl = (price - pos['ep']) * pos['sh']
            cash += pos['sh'] * price
            trades.append({'t': tk, 'pnl': pnl, 'pnl_p': (price-pos['ep'])/pos['ep']*100, 'er': 'OPEN', 'vt': pos.get('vt','?')})
            ticker_pnl[tk] += pnl
        
        final = cash
        ret = (final - capital) / capital * 100
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        tw = sum(t['pnl'] for t in wins) if wins else 0
        tl = abs(sum(t['pnl'] for t in losses)) if losses else 0.01
        pf = tw / tl
        
        combined_final += final
        all_trades.extend(trades)
        
        print(f"\n  {sn.upper()} ({sc['alloc']*100:.0f}%): ${capital:.0f} → ${final:.0f} ({ret:+.1f}%) | {len(trades)}t | {len(wins)}/{len(trades)} wins ({len(wins)/max(len(trades),1)*100:.0f}%) | PF {pf:.2f}")
        
        for er_name in ['TP','TRAIL','MAX_HOLD','STOP','OPEN']:
            et = [t for t in trades if t['er'] == er_name]
            if et:
                ep = sum(t['pnl'] for t in et)
                print(f"    {er_name:10s}: {len(et):3d} trades, ${ep:+.2f}")
        
        sorted_t = sorted(ticker_pnl.items(), key=lambda x: -x[1])
        top = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted_t[:3])
        bot = ', '.join(f"{t[0]} ${t[1]:+.0f}" for t in sorted_t[-3:])
        print(f"    Best:  {top}")
        print(f"    Worst: {bot}")
    
    cret = (combined_final - combined_capital) / combined_capital * 100
    cwins = sum(1 for t in all_trades if t['pnl'] > 0)
    cstops = sum(1 for t in all_trades if t['er'] == 'STOP')
    print(f"\n  ═══ COMBINED ═══")
    print(f"  $10,000 → ${combined_final:.0f} ({cret:+.1f}%) | {len(all_trades)} trades | {cwins} wins ({cwins/max(len(all_trades),1)*100:.0f}%) | {cstops} stops")
    
    # Per-ticker summary
    all_tpnl = defaultdict(float)
    all_tcount = defaultdict(int)
    for t in all_trades:
        all_tpnl[t['t']] += t['pnl']
        all_tcount[t['t']] += 1
    print(f"\n  Per ticker:")
    for tk, pnl in sorted(all_tpnl.items(), key=lambda x: -x[1]):
        print(f"    {tk:5s}: ${pnl:+8.2f} ({all_tcount[tk]} trades)")
    
    return cret


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
print("█" * 60)
print("  STRESS TEST — v11 on UNSEEN stocks + BEAR MARKET")
print("█" * 60)

# TEST 1: Unseen stocks (10 months)
print(f"\n{'='*60}")
print("  TEST 1: 10 Unseen Stocks (never trained on)")
print(f"{'='*60}")
print(f"  Stocks: {UNSEEN}")
print("  Fetching data from Yahoo...")

unseen_data = {}
for tk in UNSEEN:
    rows = fetch_yahoo(tk, years=4)
    if len(rows) < 300:
        print(f"  ⚠️ {tk}: only {len(rows)} days, skipping")
        continue
    unseen_data[tk] = {
        'c': [r['close'] for r in rows],
        'h': [r['high'] for r in rows],
        'l': [r['low'] for r in rows],
        'v': [r['volume'] for r in rows],
        'd': [r['date'] for r in rows],
    }
    print(f"  ✅ {tk}: {len(rows)} days")

if unseen_data:
    r1 = run_backtest(unseen_data, "Unseen 10mo", 210)

# TEST 2: Original 15 stocks during 2022 bear market
print(f"\n{'='*60}")
print("  TEST 2: Original 15 stocks — 2022 BEAR MARKET")
print(f"{'='*60}")
print("  Fetching data...")

bear_data = {}
for tk in ORIGINAL:
    rows = fetch_yahoo(tk, years=4)
    if len(rows) < 500:
        print(f"  ⚠️ {tk}: only {len(rows)} days, skipping")
        continue
    # Find 2022 window (roughly index for Jan 2022 - Dec 2022)
    dates = [r['date'] for r in rows]
    bear_data[tk] = {
        'c': [r['close'] for r in rows],
        'h': [r['high'] for r in rows],
        'l': [r['low'] for r in rows],
        'v': [r['volume'] for r in rows],
        'd': dates,
    }
    print(f"  ✅ {tk}: {len(rows)} days ({dates[0]} → {dates[-1]})")

if bear_data:
    # Find the index range for 2022
    sample_dates = list(bear_data.values())[0]['d']
    bear_start = next((i for i, d in enumerate(sample_dates) if d >= '2022-01-01'), 200)
    bear_end = next((i for i, d in enumerate(sample_dates) if d >= '2023-01-01'), len(sample_dates))
    
    # Trim data to end at Dec 2022
    for tk in bear_data:
        for key in ['c','h','l','v','d']:
            bear_data[tk][key] = bear_data[tk][key][:bear_end]
    
    print(f"\n  Bear market window: ~{sample_dates[bear_start]} → ~{sample_dates[min(bear_end-1, len(sample_dates)-1)]}")
    r2 = run_backtest(bear_data, "Bear 2022", 250)

# TEST 3: Unseen stocks during 2022 bear
print(f"\n{'='*60}")
print("  TEST 3: Unseen stocks — 2022 BEAR MARKET")
print(f"{'='*60}")

if unseen_data:
    unseen_bear = {}
    for tk in unseen_data:
        dates = unseen_data[tk]['d']
        bear_end_idx = next((i for i, d in enumerate(dates) if d >= '2023-01-01'), len(dates))
        unseen_bear[tk] = {key: unseen_data[tk][key][:bear_end_idx] for key in ['c','h','l','v','d']}
    
    r3 = run_backtest(unseen_bear, "Unseen Bear", 250)

print(f"\n{'='*60}")
print("  STRESS TEST COMPLETE")
print(f"{'='*60}")
