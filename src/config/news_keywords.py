# src/config/news_keywords.py
# ===============================
# NEWS QUERY AXES & KEYWORDS
# ===============================

# 1. Direct Bitcoin Axis (BTC)
KEYWORDS_BTC = (
    "(bitcoin OR btc OR cryptocurrency OR 'crypto price' OR 'digital gold') "
    "NOT (fraud OR scam OR hack OR theft)"
)

# 2. Technology & Infrastructure Axis (TECH)
KEYWORDS_TECH = (
    "('RAM memory' OR blockchain OR web3 OR metaverse OR 'AI investment' OR "
    "'DeFi' OR 'mining rig' OR 'semiconductor shortage')"
)

# 3. Macro / Global Markets Axis (MACRO)
KEYWORDS_MACRO = (
    "(FED OR 'interest rate' OR inflation OR recession OR "
    "'treasury bond' OR 'stock market crash' OR 'quantitative easing')"
)

KEYWORD_MAP = {
    "BTC": KEYWORDS_BTC,
    "TECH": KEYWORDS_TECH,
    "MACRO": KEYWORDS_MACRO,
}