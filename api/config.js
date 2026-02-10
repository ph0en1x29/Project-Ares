export default function handler(req, res) {
  res.setHeader('Cache-Control', 's-maxage=3600');
  res.json({
    fk: process.env.FINNHUB_KEY || '',
    sbUrl: process.env.SUPABASE_URL || 'https://ndgyyrbyboqmfnqdnvaz.supabase.co',
    sbKey: process.env.SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kZ3l5cmJ5Ym9xbWZucWRudmF6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA2ODY3ODksImV4cCI6MjA4NjI2Mjc4OX0.l3cc7DqKewHpr0Lufroo_lyMIMETYS4AB3PaYDArivs'
  });
}
