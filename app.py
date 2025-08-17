import time
import pandas as pd
import streamlit as st
import feedparser
import nltk
from typing import List, Dict


# Page & UI setup

st.set_page_config(
    page_title="AI News Sentiment",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Web Scraping + AI Sentiment")
st.caption("Scrapes RSS headlines and analyzes sentiment (choose fast VADER or Transformer).")


# Defaults 

DEFAULT_FEEDS: Dict[str, str] = {
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Reuters World": "http://feeds.reuters.com/reuters/worldNews",
    "The Hindu (Front Page)": "https://www.thehindu.com/feeder/default.rss",
    "NDTV Top Stories": "https://feeds.feedburner.com/ndtvnews-top-stories",
}


# Caching helpers

@st.cache_resource(show_spinner=False)
def ensure_vader_ready():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    from nltk.sentiment import SentimentIntensityAnalyzer  
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def load_transformer_pipeline():
    try:
        from transformers import pipeline
        nlp = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"  # small, reliable
        )
        return nlp
    except Exception as e:
        raise RuntimeError(
            "Transformers pipeline failed to load. "
            "Try the 'VADER (fast)' model or ensure 'torch' is installed."
        ) from e

@st.cache_data(show_spinner=False)
def fetch_rss(url: str, max_items: int = 30) -> pd.DataFrame:
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        rows.append({
            "source": feed.feed.get("title", "Unknown"),
            "title": entry.get("title", "").strip(),
            "link": entry.get("link", ""),
            "published": entry.get("published", entry.get("updated", "")),
        })
    df = pd.DataFrame(rows)
    # Remove empty titles & duplicates
    if not df.empty:
        df = df[df["title"].str.strip() != ""].drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df

def vader_predict(sia, texts: List[str]) -> List[Dict[str, float]]:
    out = []
    for t in texts:
        s = sia.polarity_scores(t)
        # map compound score to label
        comp = s["compound"]
        if comp >= 0.05:
            label = "POSITIVE"
        elif comp <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        out.append({"label": label, "score": abs(comp)})
    return out

def hf_predict(pipe, texts: List[str]) -> List[Dict[str, float]]:
    # Hugging Face pipeline returns list of dicts: {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
    preds = pipe(texts, truncation=True)
    # Normalize labels, add NEUTRAL if you want 3-class (we keep 2-class for HF)
    return preds

def add_sentiment(df: pd.DataFrame, model_choice: str) -> pd.DataFrame:
    if df.empty:
        return df

    titles = df["title"].tolist()

    if model_choice == "VADER (fast, no heavy install)":
        sia = ensure_vader_ready()
        preds = vader_predict(sia, titles)
    else:
        pipe = load_transformer_pipeline()
        preds = hf_predict(pipe, titles)

    df = df.copy()
    df["sentiment"] = [p["label"] for p in preds]
    df["confidence"] = [round(float(p["score"]), 4) for p in preds]
    return df


# Sidebar controls

with st.sidebar:
    st.subheader("Settings")
    sel_feeds = st.multiselect(
        "Choose RSS feeds",
        list(DEFAULT_FEEDS.keys()),
        default=["BBC World", "Reuters World", "The Hindu (Front Page)"]
    )
    max_items = st.slider("Max headlines per feed", 5, 50, 20, 5)

    model_choice = st.radio(
        "AI Model",
        ["VADER (fast, no heavy install)", "Transformer (Hugging Face)"],
        help="If Torch install fails, use VADER first."
    )

    run_btn = st.button("Scrape & Analyze", type="primary", use_container_width=True)


# Main flow....................

if run_btn:
    if not sel_feeds:
        st.warning("Pick at least one feed from the sidebar.")
        st.stop()

    with st.spinner("Fetching RSS and running AI sentiment..."):
        # scrape
        frames = []
        for k in sel_feeds:
            url = DEFAULT_FEEDS[k]
            df_feed = fetch_rss(url, max_items=max_items)
            frames.append(df_feed)
            time.sleep(0.1)  # be polite

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source", "title", "link", "published"])

        # sentiment
        df = add_sentiment(df, model_choice=model_choice)

    if df.empty:
        st.info("No items found. Try different feeds or increase the limit.")
        st.stop()

    # Filter controls
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        st.subheader("Results")
    with colB:
        uniq_sents = ["ALL"] + sorted(df["sentiment"].unique().tolist())
        sent_filter = st.selectbox("Filter by Sentiment", uniq_sents, index=0)
    with colC:
        src_filter = st.selectbox("Filter by Source", ["ALL"] + sorted(df["source"].unique().tolist()), index=0)

    filtered = df.copy()
    if sent_filter != "ALL":
        filtered = filtered[filtered["sentiment"] == sent_filter]
    if src_filter != "ALL":
        filtered = filtered[filtered["source"] == src_filter]

    # Show table
    st.dataframe(
        filtered[["source", "published", "title", "sentiment", "confidence", "link"]],
        use_container_width=True,
        hide_index=True
    )

    # Simple counts chart
    st.markdown("#### Sentiment distribution")
    counts = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.bar_chart(counts.set_index("sentiment"))

    # Download CSV
    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="news_sentiment.csv",
        mime="text/csv",
        use_container_width=True
    )

    with st.expander("ℹ️ Notes & Tips"):
        st.write(
            """
- This app uses **RSS** (friendly + stable) instead of scraping complex HTML pages.
- **VADER** is fast and light; **Transformer** gives stronger points .
- Add more feeds to `DEFAULT_FEEDS` for variety.
- Keep usage reasonable and respect publishers’ terms.
            """
        )
else:
    st.info("Pick your feeds and click **Scrape & Analyze** from the sidebar to get started.")
