// Serves real technical indicators from Alpha Vantage
// Free tier: 25 calls/day — use wisely
export default async function handler(req, res) {
  const AV_KEY = process.env.ALPHA_VANTAGE_KEY;
  if (!AV_KEY) return res.status(500).json({ error: 'No Alpha Vantage key' });
  
  const { symbol, indicator } = req.query;
  if (!symbol) return res.status(400).json({ error: 'Missing symbol' });
  
  const allowed = ['RSI', 'SMA', 'EMA', 'MACD', 'BBANDS', 'STOCH'];
  const func = (indicator || 'RSI').toUpperCase();
  if (!allowed.includes(func)) return res.status(400).json({ error: 'Invalid indicator' });
  
  // Build params based on indicator
  const params = new URLSearchParams({
    function: func,
    symbol,
    interval: 'daily',
    series_type: 'close',
    apikey: AV_KEY
  });
  
  // Add time_period for indicators that need it
  if (['RSI', 'SMA', 'EMA'].includes(func)) {
    params.set('time_period', req.query.period || (func === 'SMA' ? '200' : '14'));
  }
  
  try {
    const r = await fetch(`https://www.alphavantage.co/query?${params}`);
    const data = await r.json();
    
    // Cache for 15 minutes
    res.setHeader('Cache-Control', 's-maxage=900');
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
