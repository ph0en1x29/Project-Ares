export default function handler(req, res) {
  res.setHeader('Cache-Control', 's-maxage=3600');
  res.json({
    fk: process.env.FINNHUB_KEY || '',
    sbUrl: process.env.SUPABASE_URL || '',
    sbKey: process.env.SUPABASE_ANON_KEY || ''
  });
}
