-- ══════════════════════════════════════════════════
-- PROJECT ARES — Database Schema
-- ══════════════════════════════════════════════════

-- 1. Watchlist — user's tracked tickers
CREATE TABLE watchlist (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT,
  sector TEXT,
  added_at TIMESTAMPTZ DEFAULT NOW(),
  is_core BOOLEAN DEFAULT false,  -- core allocation vs watchlist
  allocation_pct NUMERIC DEFAULT 0,
  notes TEXT
);

-- 2. Signal Snapshots — every score computed, timestamped
CREATE TABLE signal_snapshots (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker TEXT NOT NULL,
  price NUMERIC NOT NULL,
  tech_score NUMERIC,
  value_score NUMERIC,
  catalyst_score NUMERIC,
  volatility_score NUMERIC,
  composite_score NUMERIC NOT NULL,
  signal TEXT NOT NULL,  -- STRONG BUY, BUY, ACCUMULATE, WAIT, AVOID
  rsi NUMERIC,
  pc_ratio NUMERIC,
  vix NUMERIC,
  fear_greed INTEGER,
  weights JSONB,  -- {technical:30, value:25, catalyst:25, volatility:20}
  metadata JSONB,  -- any extra data
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Paper Trades — entry/exit tracking
CREATE TABLE paper_trades (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL DEFAULT 'buy',  -- buy/sell
  entry_price NUMERIC NOT NULL,
  entry_score NUMERIC NOT NULL,
  entry_signal TEXT NOT NULL,
  entry_at TIMESTAMPTZ DEFAULT NOW(),
  exit_price NUMERIC,
  exit_score NUMERIC,
  exit_signal TEXT,
  exit_at TIMESTAMPTZ,
  shares NUMERIC DEFAULT 1,
  pnl NUMERIC,  -- calculated on exit
  pnl_pct NUMERIC,
  hold_days INTEGER,
  status TEXT DEFAULT 'open',  -- open, closed, stopped
  stop_loss NUMERIC,
  take_profit NUMERIC,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Model Accuracy — aggregated performance tracking
CREATE TABLE model_accuracy (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  period TEXT NOT NULL,  -- daily, weekly, monthly
  period_date DATE NOT NULL,
  total_signals INTEGER DEFAULT 0,
  correct_signals INTEGER DEFAULT 0,
  win_rate NUMERIC,
  avg_return_pct NUMERIC,
  best_trade_pct NUMERIC,
  worst_trade_pct NUMERIC,
  sharpe_ratio NUMERIC,
  max_drawdown_pct NUMERIC,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(period, period_date)
);

-- 5. Portfolio — current paper portfolio state
CREATE TABLE portfolio (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  starting_capital NUMERIC DEFAULT 10000,
  current_capital NUMERIC DEFAULT 10000,
  total_pnl NUMERIC DEFAULT 0,
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Initialize portfolio with $10k
INSERT INTO portfolio (starting_capital, current_capital) VALUES (10000, 10000);

-- Insert core watchlist
INSERT INTO watchlist (ticker, name, is_core, allocation_pct, notes) VALUES
  ('NVDA', 'NVIDIA', true, 44, 'Core AI play'),
  ('AMD', 'AMD', false, 0, 'Watchlist — weak Q1'),
  ('MU', 'Micron', true, 31, 'HBM4 risk, low PEG'),
  ('MSFT', 'Microsoft', true, 25, 'Azure growth, defensive'),
  ('NFLX', 'Netflix', false, 0, 'Watchlist only');

-- Indexes for performance
CREATE INDEX idx_signals_ticker_date ON signal_snapshots(ticker, created_at DESC);
CREATE INDEX idx_signals_composite ON signal_snapshots(composite_score DESC);
CREATE INDEX idx_trades_status ON paper_trades(status);
CREATE INDEX idx_trades_ticker ON paper_trades(ticker);

-- Enable RLS but allow anon access (public dashboard)
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_accuracy ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read" ON watchlist FOR SELECT USING (true);
CREATE POLICY "Public insert" ON watchlist FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update" ON watchlist FOR UPDATE USING (true);
CREATE POLICY "Public delete" ON watchlist FOR DELETE USING (true);

CREATE POLICY "Public read" ON signal_snapshots FOR SELECT USING (true);
CREATE POLICY "Public insert" ON signal_snapshots FOR INSERT WITH CHECK (true);

CREATE POLICY "Public read" ON paper_trades FOR SELECT USING (true);
CREATE POLICY "Public insert" ON paper_trades FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update" ON paper_trades FOR UPDATE USING (true);

CREATE POLICY "Public read" ON model_accuracy FOR SELECT USING (true);
CREATE POLICY "Public insert" ON model_accuracy FOR INSERT WITH CHECK (true);
CREATE POLICY "Public update" ON model_accuracy FOR UPDATE USING (true);

CREATE POLICY "Public read" ON portfolio FOR SELECT USING (true);
CREATE POLICY "Public update" ON portfolio FOR UPDATE USING (true);
