import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import feedparser
import requests
import streamlit as st
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY", "").strip()

DEFAULT_SOURCES = [
    "reuters",
    "associated-press",
    "financial-times",
    "bbc-news",
    "the-wall-street-journal",
]

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]
DEFAULT_CREATOR_VIDEOS = ["https://www.youtube.com/watch?v=W56Z0uYXVh4"]
DEFAULT_CREATOR_CHANNEL_IDS: List[str] = []
DEFAULT_MUSIC_TRACKS = [
    {
        "title": "SoundHelix Song 1 (Demo)",
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    }
]

DEMO_NEWS = [
    {
        "title": "Global markets steady as investors await earnings",
        "url": "https://example.com/news/markets-steady",
        "source": "Reuters (Demo)",
        "publishedAt": "2026-02-05T12:30:00Z",
    },
    {
        "title": "Tech shares lead gains as AI spending climbs",
        "url": "https://example.com/news/tech-gains",
        "source": "Financial Times (Demo)",
        "publishedAt": "2026-02-05T11:10:00Z",
    },
    {
        "title": "Energy prices edge higher on supply outlook",
        "url": "https://example.com/news/energy-prices",
        "source": "Associated Press (Demo)",
        "publishedAt": "2026-02-05T10:05:00Z",
    },
]

DEMO_QUOTES = [
    {
        "symbol": "AAPL",
        "price": "192.14",
        "change": "+1.05",
        "change_percent": "+0.55%",
        "latest": "2026-02-05",
    },
    {
        "symbol": "MSFT",
        "price": "415.72",
        "change": "-2.10",
        "change_percent": "-0.50%",
        "latest": "2026-02-05",
    },
    {
        "symbol": "NVDA",
        "price": "728.33",
        "change": "+8.20",
        "change_percent": "+1.14%",
        "latest": "2026-02-05",
    },
]

DEMO_SCRIPTURE = {
    "reference": "Psalm 46:1",
    "text": "God is our refuge and strength, a very present help in trouble.",
    "translation": "WEB",
}

DEMO_CREATOR_ITEMS = [
    {
        "title": "Creator Briefing: Signals and Scenarios",
        "url": "https://example.com/creators/briefing",
        "source": "Trusted Creator (Demo)",
        "published": "2026-02-05T09:00:00Z",
        "excerpt": (
            "In today’s briefing we map three macro scenarios, "
            "highlighting liquidity, energy, and policy shifts."
        ),
    }
]

st.set_page_config(page_title="News + Stocks", layout="wide")

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Cormorant+Garamond:wght@400;600&family=EB+Garamond:wght@400;600&display=swap");

    :root {
        --temple-sand: #e9dcc4;
        --temple-stone: #cbb89a;
        --temple-ink: #2c2a24;
        --temple-gold: #b9903d;
        --temple-green: #2f4f3a;
        --temple-shadow: rgba(36, 30, 21, 0.12);
    }

    .stApp {
        background: radial-gradient(circle at top, #f6efe2 0%, #efe3ce 45%, #e3d2b6 100%);
        color: var(--temple-ink);
        font-family: "EB Garamond", "Cormorant Garamond", serif;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(120deg, rgba(255,255,255,0.35) 0%, rgba(255,255,255,0.1) 60%),
            repeating-linear-gradient(
                0deg,
                rgba(201, 182, 150, 0.15) 0px,
                rgba(201, 182, 150, 0.15) 1px,
                transparent 1px,
                transparent 6px
            );
        opacity: 0.6;
        pointer-events: none;
        z-index: 0;
    }

    .block-container {
        padding-top: 2rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4 {
        font-family: "Cinzel", serif;
        letter-spacing: 0.08em;
        color: var(--temple-green);
    }

    h1 {
        font-size: 2.4rem;
        text-transform: uppercase;
        border-bottom: 3px solid var(--temple-gold);
        padding-bottom: 0.4rem;
    }

    .temple-panel {
        background: linear-gradient(160deg, rgba(255, 255, 255, 0.7) 0%, rgba(245, 235, 214, 0.9) 100%);
        border: 1px solid rgba(164, 138, 90, 0.45);
        box-shadow: 0 12px 30px var(--temple-shadow);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.5rem;
        position: relative;
    }

    .temple-panel::before {
        content: "";
        position: absolute;
        top: -10px;
        left: 18px;
        right: 18px;
        height: 10px;
        background: linear-gradient(90deg, transparent, var(--temple-gold), transparent);
        opacity: 0.7;
    }

    .temple-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--temple-gold), transparent);
        margin: 1rem 0 1.2rem;
        border: none;
    }

    .stCaption, .stMarkdown small {
        color: rgba(44, 42, 36, 0.7);
    }

    .stAudio, .stInfo, .stWarning, .stError {
        border-radius: 12px;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: rgba(255, 252, 246, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(164, 138, 90, 0.35);
    }

    .stSlider > div {
        color: var(--temple-green);
    }

    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--temple-gold), transparent);
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("News + Stocks Dashboard")
st.caption("News on the left, stocks on the right. Built with Streamlit.")

with st.sidebar:
    st.header("Configuration")
    demo_mode = st.checkbox(
        "Demo mode (no API keys required)",
        value=not (NEWSAPI_KEY and ALPHAVANTAGE_KEY),
    )
    enable_scripture = st.checkbox("Show daily scripture", value=True)
    scripture_translation = st.text_input(
        "Bible translation ID",
        value="web",
        help="Used with bible-api.com (e.g., web, kjv).",
    )
    news_sources = st.multiselect(
        "News sources",
        options=DEFAULT_SOURCES,
        default=DEFAULT_SOURCES,
        help="Requires a NewsAPI key with access to the selected sources.",
    )
    symbols_input = st.text_input(
        "Stock symbols (comma-separated)",
        value=", ".join(DEFAULT_SYMBOLS),
    )
    refresh_seconds = st.slider("Refresh (seconds)", 30, 600, 120, 30)

    st.markdown("---")
    st.subheader("Creator panel")
    enable_creator_panel = st.checkbox("Enable creator panel", value=True)
    max_creator_videos = st.slider("Videos per channel", 1, 5, 1, 1)
    creator_channels_input = st.text_input(
        "YouTube channel IDs (comma-separated)",
        value=", ".join(DEFAULT_CREATOR_CHANNEL_IDS),
        help="Paste channel IDs (e.g., UCxxxx). Channel URLs with /channel/UCxxxx also work.",
    )
    creator_videos_input = st.text_area(
        "YouTube video URLs (one per line or comma-separated)",
        value="\n".join(DEFAULT_CREATOR_VIDEOS),
        height=90,
    )

    st.markdown("---")
    st.subheader("Music")
    enable_music = st.checkbox("Enable music player", value=True)
    music_tracks_input = st.text_area(
        "Audio tracks (one per line, Title | URL)",
        value="\n".join(f"{item['title']} | {item['url']}" for item in DEFAULT_MUSIC_TRACKS),
        height=90,
    )

    st.markdown("---")
    st.subheader("API status")
    st.write("NewsAPI key:", "✅" if NEWSAPI_KEY else "⚠️ missing")
    st.write("Alpha Vantage key:", "✅" if ALPHAVANTAGE_KEY else "⚠️ missing")

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
creator_channel_values = [
    value.strip() for value in creator_channels_input.split(",") if value.strip()
]
creator_video_values = [
    value.strip()
    for value in creator_videos_input.replace(",", "\n").splitlines()
    if value.strip()
]
music_tracks_values = [
    value.strip() for value in music_tracks_input.splitlines() if value.strip()
]


@st.cache_data(ttl=300)
def fetch_news(sources: List[str], demo: bool) -> List[Dict[str, str]]:
    if demo:
        return DEMO_NEWS
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "sources": ",".join(sources),
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    results = []
    for article in articles:
        results.append(
            {
                "title": article.get("title", "(No title)"),
                "url": article.get("url", ""),
                "source": (article.get("source") or {}).get("name", ""),
                "publishedAt": article.get("publishedAt", ""),
            }
        )
    return results


@st.cache_data(ttl=120)
def fetch_quote(symbol: str, demo: bool) -> Optional[Dict[str, str]]:
    if demo:
        for quote in DEMO_QUOTES:
            if quote["symbol"] == symbol:
                return quote
        return None
    if not ALPHAVANTAGE_KEY:
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_KEY,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("Global Quote", {})
    if not data:
        return None

    return {
        "symbol": data.get("01. symbol", symbol),
        "price": data.get("05. price", ""),
        "change": data.get("09. change", ""),
        "change_percent": data.get("10. change percent", ""),
        "latest": data.get("07. latest trading day", ""),
    }


@st.cache_data(ttl=86400)
def fetch_daily_verse(
    translation_id: str, date_key: str, demo: bool
) -> Optional[Dict[str, str]]:
    if demo:
        return DEMO_SCRIPTURE
    if not translation_id:
        return None

    url = f"https://bible-api.com/data/{translation_id}/random"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    verse = data.get("random_verse", {})
    translation = data.get("translation", {}).get("identifier", translation_id)
    book = verse.get("book")
    chapter = verse.get("chapter")
    chapter_str = str(chapter) if chapter is not None else ""
    reference = " ".join(part for part in [book, chapter_str] if part)
    verse_number = verse.get("verse")
    if verse_number is not None:
        reference = f"{reference}:{verse_number}".strip()
    return {
        "reference": reference.strip(),
        "text": (verse.get("text") or "").strip(),
        "translation": translation,
    }


def parse_video_id(value: str) -> Optional[str]:
    if not value:
        return None
    if len(value) == 11 and "/" not in value and "?" not in value:
        return value
    try:
        parsed = urlparse(value)
        if parsed.netloc.endswith("youtube.com"):
            query = parse_qs(parsed.query)
            return (query.get("v") or [None])[0]
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.lstrip("/")
    except ValueError:
        return None
    return None


def parse_channel_id(value: str) -> Optional[str]:
    if not value:
        return None
    if value.startswith("UC") and len(value) >= 10:
        return value
    try:
        parsed = urlparse(value)
        if "/channel/" in parsed.path:
            return parsed.path.split("/channel/")[-1].split("/")[0]
    except ValueError:
        return None
    return None


@st.cache_data(ttl=900)
def fetch_latest_videos(channel_ids: List[str], max_per_channel: int) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for channel_id in channel_ids:
        feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:max_per_channel]:
            items.append(
                {
                    "title": entry.get("title", "Untitled"),
                    "url": entry.get("link", ""),
                    "source": entry.get("author", "YouTube"),
                    "published": entry.get("published", ""),
                    "video_id": entry.get("yt_videoid") or parse_video_id(entry.get("link", "")),
                }
            )
    return items


@st.cache_data(ttl=900)
def fetch_transcript_excerpt(video_id: str, max_chars: int = 800) -> Tuple[Optional[str], Optional[str]]:
    if not video_id:
        return None, "Missing video ID."
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        try:
            transcript = transcripts.find_transcript(["en"])
        except NoTranscriptFound:
            try:
                transcript = transcripts.find_generated_transcript(["en"])
            except NoTranscriptFound:
                transcript = next(iter(transcripts))
        data = transcript.fetch()
        text = " ".join(chunk.get("text", "") for chunk in data)
        text = " ".join(text.split())
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "…"
        return text, None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Transcript error: {exc}"


def format_published_time(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%b %d, %Y %H:%M UTC")
    except ValueError:
        return value


def parse_music_tracks(lines: List[str]) -> List[Dict[str, str]]:
    tracks = []
    title_counts: Dict[str, int] = {}
    for line in lines:
        if "|" in line:
            title, url = [part.strip() for part in line.split("|", 1)]
        else:
            title, url = "Track", line.strip()
        if url:
            clean_title = title or "Track"
            title_counts[clean_title] = title_counts.get(clean_title, 0) + 1
            if title_counts[clean_title] > 1:
                clean_title = f"{clean_title} {title_counts[clean_title]}"
            tracks.append({"title": clean_title, "url": url})
    return tracks


col_news, col_stocks = st.columns([2, 1], gap="large")


def panel_start(title: str) -> None:
    st.markdown('<div class="temple-panel">', unsafe_allow_html=True)
    st.subheader(title)


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def temple_divider() -> None:
    st.markdown('<div class="temple-divider"></div>', unsafe_allow_html=True)


with col_news:
    if enable_scripture:
        panel_start("Daily Scripture")
        today_key = date.today().isoformat()
        try:
            verse = fetch_daily_verse(
                scripture_translation.strip().lower(), today_key, demo_mode
            )
        except requests.RequestException as exc:
            verse = None
            st.error(f"Bible API error: {exc}")

        if verse and verse.get("text"):
            st.write(verse["text"])
            ref = verse.get("reference", "").strip()
            translation = verse.get("translation", "").upper()
            caption_parts = [part for part in [ref, translation] if part]
            if caption_parts:
                st.caption(" • ".join(caption_parts))
        else:
            st.info("No verse available. Check the translation ID or try again.")
        panel_end()

    panel_start("News")
    if demo_mode:
        st.info("Demo mode is on. Showing sample headlines.")
    elif not NEWSAPI_KEY:
        st.info("Add `NEWSAPI_KEY` to load real headlines.")
    elif not news_sources:
        st.warning("Choose at least one news source.")
    else:
        with st.spinner("Loading headlines..."):
            try:
                news_items = fetch_news(news_sources, demo_mode)
            except requests.RequestException as exc:
                st.error(f"News API error: {exc}")
                news_items = []

        if not news_items:
            st.warning("No headlines found. Check sources or API access.")
        for item in news_items:
            title = item["title"]
            url = item["url"]
            source = item["source"]
            published = format_published_time(item["publishedAt"])
            meta = " • ".join(part for part in [source, published] if part)
            st.markdown(f"**[{title}]({url})**")
            if meta:
                st.caption(meta)
            temple_divider()
    panel_end()

    if enable_creator_panel:
        panel_start("End Days Panel (Creators)")
        st.caption("Latest videos and transcript excerpts from trusted creators.")

        creator_items: List[Dict[str, str]] = []
        channel_ids = [cid for cid in (parse_channel_id(v) for v in creator_channel_values) if cid]
        if channel_ids:
            try:
                creator_items.extend(fetch_latest_videos(channel_ids, max_creator_videos))
            except Exception as exc:
                st.error(f"Creator feed error: {exc}")

        for video_url in creator_video_values:
            video_id = parse_video_id(video_url)
            creator_items.append(
                {
                    "title": video_id or "Manual video",
                    "url": video_url,
                    "source": "Manual",
                    "published": "",
                    "video_id": video_id,
                }
            )

        if demo_mode and not creator_items:
            creator_items = DEMO_CREATOR_ITEMS.copy()

        if not creator_items:
            st.info("Add channel IDs or video URLs to populate this panel.")
        else:
            for item in creator_items:
                title = item.get("title") or "Untitled"
                url = item.get("url", "")
                source = item.get("source", "")
                published = format_published_time(item.get("published", ""))

                if url:
                    st.markdown(f"**[{title}]({url})**")
                else:
                    st.markdown(f"**{title}**")
                meta = " • ".join(part for part in [source, published] if part)
                if meta:
                    st.caption(meta)

                video_id = item.get("video_id")
                if video_id:
                    excerpt, error = fetch_transcript_excerpt(video_id)
                    if excerpt:
                        st.write(excerpt)
                    elif error:
                        st.caption(f"Transcript unavailable: {error}")
                elif item.get("excerpt"):
                    st.write(item["excerpt"])
                temple_divider()
        panel_end()

with col_stocks:
    panel_start("Stocks")
    if demo_mode:
        st.info("Demo mode is on. Showing sample quotes.")
    elif not ALPHAVANTAGE_KEY:
        st.info("Add `ALPHAVANTAGE_KEY` to load real quotes.")
    elif not symbols:
        st.warning("Enter at least one stock symbol.")
    else:
        with st.spinner("Loading quotes..."):
            quotes = []
            for symbol in symbols:
                try:
                    quote = fetch_quote(symbol, demo_mode)
                except requests.RequestException as exc:
                    st.error(f"Stock API error for {symbol}: {exc}")
                    quote = None
                if quote:
                    quotes.append(quote)
                time.sleep(0.15)

        if not quotes:
            st.warning("No quotes returned. Check symbols or API access.")
        for quote in quotes:
            st.markdown(f"**{quote['symbol']}**")
            st.write(f"Price: {quote['price']}")
            st.write(f"Change: {quote['change']} ({quote['change_percent']})")
            if quote["latest"]:
                st.caption(f"Latest trading day: {quote['latest']}")
            temple_divider()
    panel_end()

    if enable_music:
        panel_start("Music")
        tracks = parse_music_tracks(music_tracks_values)
        if demo_mode and not tracks:
            tracks = DEFAULT_MUSIC_TRACKS.copy()
        if not tracks:
            st.info("Add audio track URLs in the sidebar to enable playback.")
        else:
            track_titles = [track["title"] for track in tracks]
            selected_title = st.selectbox("Track", track_titles)
            selected = next(track for track in tracks if track["title"] == selected_title)
            st.audio(selected["url"])
            if "soundhelix.com" in selected["url"]:
                st.caption("Demo audio courtesy of SoundHelix (credit required).")
        panel_end()

st.caption(
    "Note: Access to premium sources like Reuters/AP/FT/WSJ depends on your provider plan."
)
