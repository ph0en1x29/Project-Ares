#!/bin/bash
# Fetch all data first, then run Python backtest
SB_URL="https://ndgyyrbyboqmfnqdnvaz.supabase.co"
SB_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kZ3l5cmJ5Ym9xbWZucWRudmF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA2ODY3ODksImV4cCI6MjA4NjI2Mjc4OX0.l3cc7DqKewHpr0Lufroo_lyMIMETYS4AB3PaYDArivs"

mkdir -p /tmp/ares-data

TICKERS="NVDA AMD MU MSFT AAPL AMZN GOOGL META AVGO QCOM INTC TSM MRVL LRCX AMAT"

echo "Fetching data..."
for T in $TICKERS; do
  curl -s "$SB_URL/rest/v1/daily_prices?ticker=eq.$T&order=date.asc&limit=2000" \
    -H "apikey: $SB_KEY" -H "Authorization: Bearer $SB_KEY" > /tmp/ares-data/$T.json
  ROWS=$(python3 -c "import json; print(len(json.load(open('/tmp/ares-data/$T.json'))))")
  echo "  $T: $ROWS days"
done

echo "Running backtest..."
python3 /home/jay/entry-signal-model/backtest-v11-local.py
