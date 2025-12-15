import streamlit as st
import pandas as pd
import scrapetube
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
import itertools
import re
import time
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Social Media Intelligence Dashboard", page_icon="📈", layout="wide")

# --- Custom CSS for Branding Footer ---
st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #31333F;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e6e6ea;
        z-index: 100;
    }
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        font-weight: bold;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    /* Add padding to bottom of main container so footer doesn't cover content */
    .block-container {
        padding-bottom: 80px;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching Resources ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

setup_nltk()

# --- Helper Functions ---

def parse_relative_date(time_str):
    """Converts '2 weeks ago' into datetime objects."""
    if not time_str: return datetime.now()
    now = datetime.now()
    time_str = str(time_str).lower()
    
    try:
        if 'second' in time_str: delta = timedelta(seconds=int(re.search(r'\d+', time_str).group()))
        elif 'minute' in time_str: delta = timedelta(minutes=int(re.search(r'\d+', time_str).group()))
        elif 'hour' in time_str: delta = timedelta(hours=int(re.search(r'\d+', time_str).group()))
        elif 'day' in time_str: delta = timedelta(days=int(re.search(r'\d+', time_str).group()))
        elif 'week' in time_str: delta = timedelta(weeks=int(re.search(r'\d+', time_str).group()))
        elif 'month' in time_str: delta = timedelta(days=int(re.search(r'\d+', time_str).group()) * 30)
        elif 'year' in time_str: delta = timedelta(days=int(re.search(r'\d+', time_str).group()) * 365)
        else: delta = timedelta(seconds=0)
        return now - delta
    except:
        return now

def parse_votes(vote_str):
    if not vote_str: return 0
    s = str(vote_str).upper().strip()
    if 'K' in s: return int(float(re.sub(r'[^0-9.]', '', s)) * 1000)
    elif 'M' in s: return int(float(re.sub(r'[^0-9.]', '', s)) * 1000000)
    else:
        clean_s = re.sub(r'[^0-9]', '', s)
        return int(clean_s) if clean_s else 0

def get_sentiment(text, sia):
    if not text: return "Neutral", 0.0
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05: return "Positive", score
    elif score <= -0.05: return "Negative", score
    else: return "Neutral", score

def extract_phrases(text_series, n=2):
    text = text_series.dropna().astype(str).tolist()
    if not text:
        return pd.DataFrame(columns=['Phrase', 'Count'])

    vec = CountVectorizer(ngram_range=(n, n), stop_words='english', min_df=2)
    try:
        bag_of_words = vec.fit_transform(text)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq[:20], columns=['Phrase', 'Count'])
    except ValueError:
        return pd.DataFrame(columns=['Phrase', 'Count'])

# --- Scraper ---
def scrape_data(channel_url, video_limit, comment_limit, sort_mode):
    downloader = YoutubeCommentDownloader()
    sia = SentimentIntensityAnalyzer()
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    sort_map = {"Latest": "newest", "Popular": "popular", "Oldest": "oldest"}
    
    try:
        videos = list(scrapetube.get_channel(channel_url=channel_url, limit=video_limit, sort_by=sort_map[sort_mode]))
    except Exception as e:
        st.error(f"Error fetching channel: {e}")
        return pd.DataFrame()

    all_comments = []
    total_videos = len(videos)
    
    for i, video in enumerate(videos):
        video_id = video['videoId']
        try: title = video['title']['runs'][0]['text']
        except: title = "Unknown Title"
            
        status_text.text(f"Scanning ({i+1}/{total_videos}): {title[:50]}...")
        progress_bar.progress((i + 1) / total_videos)
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        try:
            generator = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
            batch = list(itertools.islice(generator, comment_limit))
            
            for c in batch:
                text = c.get('text', '')
                sentiment_label, sentiment_score = get_sentiment(text, sia)
                clean_votes = parse_votes(c.get('votes', '0'))
                approx_date = parse_relative_date(c.get('time', ''))

                all_comments.append({
                    "Video Title": title,
                    "Author": c.get('author', 'Anonymous'),
                    "Comment": text,
                    "Likes": clean_votes,
                    "Sentiment": sentiment_label,
                    "Sentiment Score": sentiment_score,
                    "Date": approx_date,
                    "Relative Time": c.get('time', 'Just now'),
                    "Video URL": video_url
                })
        except Exception as e:
            continue

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(all_comments)

# --- UI Layout ---

st.sidebar.title("🔍 Configuration")
url_input = st.sidebar.text_input("Channel URL", "https://www.youtube.com/@ThePrimeagen")
sort_mode = st.sidebar.selectbox("Sort Videos By", ["Latest", "Popular", "Oldest"])
video_limit = st.sidebar.slider("Videos to Analyze", 1, 50, 5)
comment_limit = st.sidebar.slider("Comments / Video", 20, 1000, 50)
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Generate Intelligence Report", type="primary")

st.title("🧠 Social Media Intelligence Report")
st.markdown("Advanced analytics for community sentiment, emerging trends, and engagement.")

if run_btn:
    with st.spinner('Gathering Intelligence...'):
        df = scrape_data(url_input, video_limit, comment_limit, sort_mode)
        st.session_state['data'] = df

if 'data' in st.session_state and not st.session_state['data'].empty:
    df = st.session_state['data']
    
    # --- GLOBAL FILTERS ---
    st.markdown("### 🕵️ Data Filters")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_keyword = st.text_input("Filter by Keyword (e.g., 'price', 'bug', 'love')")
    with col_f2:
        filter_sent = st.multiselect("Filter Sentiment", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
    
    # Apply Filters
    filtered_df = df[df['Sentiment'].isin(filter_sent)]
    if filter_keyword:
        filtered_df = filtered_df[filtered_df['Comment'].str.contains(filter_keyword, case=False, na=False)]
    
    st.divider()

    # --- KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume", f"{len(filtered_df)} Comments")
    
    if len(filtered_df) > 0:
        pos_pct = len(filtered_df[filtered_df['Sentiment']=='Positive']) / len(filtered_df) * 100
        neg_pct = len(filtered_df[filtered_df['Sentiment']=='Negative']) / len(filtered_df) * 100
        nss = pos_pct - neg_pct
    else:
        nss = 0
        
    c2.metric("Net Sentiment Score", f"{nss:.1f}", delta=f"{nss:.1f}")
    c3.metric("Engagement (Likes)", f"{filtered_df['Likes'].sum():,}")
    c4.metric("Active Authors", filtered_df['Author'].nunique())

    # --- TABS ---
    tab_trend, tab_topic, tab_video, tab_raw = st.tabs(["📈 Trends over Time", "🗣️ Topics & Phrases", "📹 Video Performance", "📄 Data Explorer"])

    with tab_trend:
        st.subheader("Sentiment Timeline")
        if not filtered_df.empty:
            # Prepare data
            timeline_df = filtered_df.set_index('Date').resample('D')['Sentiment Score'].mean().reset_index()
            
            # IMPROVEMENT: Add Rolling Average to smooth out the jagged lines
            timeline_df['Smoothed'] = timeline_df['Sentiment Score'].rolling(window=3, min_periods=1).mean()

            fig_trend = go.Figure()
            
            # Raw Data (faint)
            fig_trend.add_trace(go.Scatter(
                x=timeline_df['Date'], y=timeline_df['Sentiment Score'],
                mode='lines', name='Daily Score',
                line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot')
            ))
            
            # Smoothed Trend (bold)
            fig_trend.add_trace(go.Scatter(
                x=timeline_df['Date'], y=timeline_df['Smoothed'],
                mode='lines+markers', name='Trend (3-Day Avg)',
                line=dict(color='#FF4B4B', width=3)
            ))

            fig_trend.update_layout(
                title="Sentiment Evolution (Smoothed)",
                xaxis_title="Date",
                yaxis_title="Sentiment Score (-1 to +1)",
                yaxis=dict(range=[-1, 1]),
                hovermode="x unified",
                template="plotly_white"
            )
            
            # Add visual zones
            fig_trend.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.05, layer="below", line_width=0)
            fig_trend.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.05, layer="below", line_width=0)

            st.plotly_chart(fig_trend, use_container_width=True)
            st.info("ℹ️ **Interpretation:** The Red line shows the moving average. Points in the green zone are generally positive; red zone are negative.")
        else:
            st.warning("No data to plot.")

    with tab_topic:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top Discussed Phrases")
            phrase_df = extract_phrases(filtered_df['Comment'], n=2)
            
            if not phrase_df.empty:
                fig_bar = px.bar(phrase_df, x='Count', y='Phrase', orientation='h', 
                                 title="Common 2-Word Phrases",
                                 color='Count', color_continuous_scale='Viridis')
                fig_bar.update_layout(yaxis=dict(autorange="reversed")) 
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("Not enough text data to extract phrases.")
            
        with c2:
            st.subheader("Sentiment Distribution")
            if not filtered_df.empty:
                fig_pie = px.pie(filtered_df, names='Sentiment', color='Sentiment', 
                                 hole=0.4, title="Share of Voice",
                                 color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab_video:
        st.subheader("Video Performance Matrix")
        if not filtered_df.empty:
            vid_stats = filtered_df.groupby('Video Title').agg({
                'Sentiment Score': 'mean',
                'Likes': 'sum',
                'Comment': 'count'
            }).reset_index().rename(columns={'Comment': 'Comment Count'})
            
            # IMPROVEMENT: Use Quadrants logic
            fig_bub = px.scatter(vid_stats, x='Sentiment Score', y='Likes', size='Comment Count', hover_name='Video Title',
                                 color='Sentiment Score', color_continuous_scale='RdBu', size_max=60,
                                 title="Engagement vs. Sentiment (Bubble Size = Comment Volume)")
            
            fig_bub.add_vline(x=0, line_dash="dash", line_color="gray")
            
            # Annotations for Quadrants
            max_y = vid_stats['Likes'].max() * 0.9
            fig_bub.add_annotation(x=0.8, y=max_y, text="🔥 Viral Hits", showarrow=False, font=dict(color="green", size=14))
            fig_bub.add_annotation(x=-0.8, y=max_y, text="⚠️ Controversial", showarrow=False, font=dict(color="red", size=14))
            
            st.plotly_chart(fig_bub, use_container_width=True)
            st.markdown("""
            **How to read this chart:**
            * **Top Right (Viral Hits):** High Engagement, Positive Feedback.
            * **Top Left (Controversial):** High Engagement, but Negative Feedback.
            * **Bottom Right (Niche):** Low Engagement, but people like it.
            """)

    with tab_raw:
        st.dataframe(filtered_df[['Date', 'Author', 'Comment', 'Likes', 'Sentiment', 'Video Title']], use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Filtered Data to CSV", csv, "social_intelligence_report.csv", "text/csv")

elif run_btn:
    st.warning("No data found.")
else:
    st.info("👈 Enter a URL and settings in the sidebar to begin.")

# --- FOOTER (Theme-Adaptive & Clean) ---
st.divider()  # Adds a subtle visual separator

st.markdown("""
<style>
    .footer-container {
        width: 100%;
        text-align: center;
        padding-top: 30px;
        padding-bottom: 30px;
        font-size: 14px;
        opacity: 0.7; /* Makes it subtle/professional */
    }
    .footer-container a {
        color: #FF4B4B; /* Streamlit Red matches standard links */
        text-decoration: none;
        font-weight: 600;
        transition: 0.3s;
        margin: 0 10px;
    }
    .footer-container a:hover {
        opacity: 0.8;
        text-decoration: underline;
    }
</style>

<div class="footer-container">
    <p>Made with ❤️ by <b>Saif Ali</b></p>
    <p>
        <a href="https://www.sudobotz.com" target="_blank">🌐 sudobotz.com</a> • 
        <a href="mailto:contact@sudobotz.com">📧 contact@sudobotz.com</a> • 
        <a href="https://fiverr.com/saifalimz" target="_blank">💼 Hire me on Fiverr</a>
    </p>
</div>
""", unsafe_allow_html=True)