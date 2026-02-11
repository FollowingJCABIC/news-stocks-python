# Project Skeleton: News + Stocks + Creator Signals

## Vision
A focused dashboard that keeps trusted news sources on the left, market data on the right, and a third signal panel that surfaces narratives from trusted creators (e.g., YouTubers with transcripts). The goal is to reduce noise while preserving breadth.

## Current State (Implemented)
- Streamlit UI with two columns:
  - News panel (left)
  - Stocks panel (right)
- Daily scripture panel (Bible API)
- Music player panel (audio URLs)
- Demo mode for no‑key preview
- Creator panel (in the News column) that:
  - Pulls latest uploads by channel RSS
  - Fetches transcript excerpts when captions exist

## MVP Scope (Next Stable Baseline)
- One unified layout with three sections:
  - News (trusted sources)
  - Stocks (watchlist)
  - Creator Signals (“End Days Panel”)
- Configurable watchlist and sources
- Local caching to limit API calls
- Simple performance guardrails (timeouts, error messaging)

## Near‑Term Roadmap (Skeleton)
1. **Data reliability**
   - Validate transcript availability; graceful fallbacks
   - Add retry/backoff and short‑term cache

2. **Creator pipeline v1**
   - Creator list management (IDs, handles)
   - Multi‑video scan per creator
   - Lightweight transcript summary (bullets)

3. **Creator pipeline v2 (optional)**
   - Whisper‑based transcription for videos without captions
   - Relevance scoring to rank stories
   - Topic tags / entity extraction

4. **Trusted sources expansion**
   - Swap in enterprise feeds (Reuters/AP/FT/WSJ) as licensing allows
   - Support multiple provider adapters

## Architecture (High‑Level)
- **UI:** Streamlit for rapid iteration
- **News Provider:** Adapter function (NewsAPI now; enterprise later)
- **Stocks Provider:** Adapter function (Alpha Vantage now; enterprise later)
- **Creator Pipeline:**
  - Discovery: YouTube RSS (no key) or Data API (key)
  - Transcript: `youtube-transcript-api`
  - Summarizer: placeholder for later

## Data Model (Conceptual)
- `NewsItem`: title, url, source, publishedAt
- `StockQuote`: symbol, price, change, change_percent, latest
- `CreatorItem`: title, url, channel, published, transcript_excerpt, score

## Non‑Goals (For Now)
- Full social feed ingestion
- Personalized ranking algorithms
- Automated trading decisions

## Open Questions
- Which enterprise news provider will be licensed first?
- How many creators and how often should we refresh?
- What “story” format feels most useful: bullets, TL;DR, or key quotes?

## Notes on Ethics & Licensing
- Only use licensed APIs and permitted sources.
- Avoid scraping or redistributing restricted content.
- Keep transcripts within fair‑use excerpts unless licensed.

## Next Decisions Needed
- Pick a provider for news and market data
- Decide creator list and refresh cadence
- Decide summarization strategy (none vs. rules‑based vs. model)
