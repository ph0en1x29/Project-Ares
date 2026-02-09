# Entry Signal Model

**Live tech stock entry signal scoring engine** — multi-factor model with real-time data.

## What It Does

Scores 5 tech stocks (NVDA, AMD, MU, MSFT, NFLX) across 4 weighted factors to generate entry signals:

| Factor | Weight | Sources |
|--------|--------|---------|
| **Technical** | 30% | RSI(14), SMA50/200, support/resistance |
| **Value** | 25% | ATH drawdown, PEG ratio, analyst consensus |
| **Catalyst** | 25% | Earnings proximity, importance, sentiment |
| **Volatility** | 20% | VIX, IV percentile, Fear/Greed Index |

### Signal Levels
- **STRONG BUY** (75+) — Multiple factors aligned
- **BUY** (60+) — Good entry opportunity
- **ACCUMULATE** (45+) — Dollar-cost average
- **WAIT** (30+) — Hold off
- **AVOID** (<30) — Stay away

## Data Sources

- **Yahoo Finance** — Price polling (~15min delay)
- **Finnhub WebSocket** — Real-time prices (free API key)
- **Historical RSI** — Calculated from 90 days of Yahoo closes
- **Fear/Greed Index** — Live from alternative.me

## Setup

Just open `index.html` in a browser. No build step, no dependencies.

For real-time prices: click "Add Finnhub Key" → paste free key from [finnhub.io](https://finnhub.io).

## Roadmap

- [ ] Supabase backend (paper trading, signal history)
- [ ] Options flow data
- [ ] Sector momentum (relative strength)
- [ ] Macro/Fed calendar integration
- [ ] News sentiment (NLP)
- [ ] Backtesting engine
- [ ] AI trader (Phoenix) with paper portfolio

## Disclaimer

⚠️ Decision-support tool only. Not financial advice.
