export default function handler(req, res) {
  // Serves Finnhub key from Vercel env var — never in repo
  res.setHeader('Cache-Control', 's-maxage=3600');
  res.json({ fk: process.env.FINNHUB_KEY || '' });
}
