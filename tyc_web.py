import streamlit as st
import pandas as pd
import scrapetube
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import google.generativeai as genai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="Social Media Intelligence Dashboard", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Branding & UI ---
st.markdown("""
<style>
    /* Main container spacing to prevent footer overlap */
    .block-container { padding-bottom: 100px; }
    
    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Button Styling */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 600;
    }

    /* Footer Styling - Professional & Clean */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.95); /* Glass effect */
        backdrop-filter: blur(10px);
        color: #4b5563;
        text-align: center;
        padding: 15px 0;
        font-size: 14px;
        border-top: 1px solid #e5e7eb;
        z-index: 9999;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .footer p { margin: 5px 0; }
    
    .footer a {
        color: #FF4B4B;
        text-decoration: none;
        font-weight: 600;
        margin: 0 12px;
        transition: all 0.2s ease;
    }
    
    .footer a:hover {
        color: #D43636;
        text-decoration: underline;
    }
    
    .highlight { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
DEFAULT_API_KEY = "AIzaSyAbhpK36plAJZ-vLJ8ed97RB9Y6QcRtNRA" # ⚠️ Replace if revoked

# --- Caching Resources ---
@st.cache_resource
def setup_nltk():
    resources = ['vader_lexicon', 'stopwords']
    for r in resources:
        try:
            nltk.data.find(f'corpora/{r}')
        except LookupError:
            nltk.download(r, quiet=True)

setup_nltk()

# --- AI Helper (Bulletproof Version) ---
def get_ai_response(comment_text, api_key):
    """Generates a response, automatically trying different free-tier models."""
    if not api_key: return "Error: No API Key provided."
    
    genai.configure(api_key=api_key)
    
    # Priority list for Free Tier users
    model_candidates = [
        'gemini-1.5-flash',       
        'gemini-flash-latest',    
        'gemini-1.5-pro',        
        'gemini-pro',             
        'gemini-2.0-flash-exp',   
    ]
    
    last_error = ""

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            
            # STRICT PROMPT to prevent character counts
            prompt = f"""
            You are a professional Social Media Manager. 
            Write a polite, empathetic, and professional reply to the following YouTube comment.
            
            - If it is a complaint, apologize and offer a solution.
            - If it is hate speech, write "Ignore/Hide".
            - If it is praise, thank them warmly.
            - The response MUST be under 280 characters.
            - DO NOT include the character count (e.g., "148 chars") or any explanations.
            - Return ONLY the clean text of the reply.
            
            Comment: "{comment_text}"
            """
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            last_error = str(e)
            continue

    return f"❌ AI Engine Error: {last_error}"

# --- Helper Functions ---
def parse_relative_date(time_str):
    if not time_str: return datetime.now()
    now = datetime.now()
    s = str(time_str).lower().replace(" (edited)", "").strip()
    try: val = int(re.search(r'\d+', s).group())
    except: val = 0

    if 'second' in s: delta = timedelta(seconds=val)
    elif 'minute' in s: delta = timedelta(minutes=val)
    elif 'hour' in s: delta = timedelta(hours=val)
    elif 'day' in s: delta = timedelta(days=val)
    elif 'week' in s: delta = timedelta(weeks=val)
    elif 'month' in s: delta = timedelta(days=val * 30)
    elif 'year' in s: delta = timedelta(days=val * 365)
    else: delta = timedelta(seconds=0)
    return now - delta

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

def extract_topics(text_series, n=2):
    text = text_series.dropna().astype(str).tolist()
    if not text: return pd.DataFrame()
    custom_stops = list(stopwords.words('english')) + ['video', 'youtube', 'channel', 'content', 'subscribe', 'watching', 'guy', 'bro', 'like']
    vec = CountVectorizer(ngram_range=(n, n), stop_words=custom_stops, min_df=2, max_features=50)
    try:
        bag_of_words = vec.fit_transform(text)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return pd.DataFrame(sorted(words_freq, key=lambda x: x[1], reverse=True)[:20], columns=['Phrase', 'Count'])
    except ValueError:
        return pd.DataFrame()

# --- Scraper ---
def scrape_data(channel_url, video_limit, comment_limit, sort_mode):
    downloader = YoutubeCommentDownloader()
    sia = SentimentIntensityAnalyzer()
    sort_map = {"Latest": "newest", "Popular": "popular"}
    
    try:
        videos = list(scrapetube.get_channel(channel_url=channel_url, limit=video_limit, sort_by=sort_map[sort_mode]))
    except Exception as e:
        st.error(f"Error fetching channel: {e}")
        return pd.DataFrame()

    all_comments = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, video in enumerate(videos):
        try:
            title = video['title']['runs'][0]['text']
            video_id = video['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            status_text.text(f"Scanning ({i+1}/{len(videos)}): {title[:40]}...")
            progress_bar.progress((i + 1) / len(videos))
            
            generator = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
            batch = []
            try:
                for _ in range(comment_limit): batch.append(next(generator))
            except: pass

            for c in batch:
                text = c.get('text', '')
                sent_label, sent_score = get_sentiment(text, sia)
                date_obj = parse_relative_date(c.get('time', ''))
                
                all_comments.append({
                    "Video Title": title,
                    "Author": c.get('author', 'Anonymous'),
                    "Comment": text,
                    "Likes": parse_votes(c.get('votes', '0')),
                    "Sentiment": sent_label,
                    "Sentiment Score": sent_score,
                    "Date": date_obj,
                    "Video URL": video_url,
                    "cid": c.get('cid'),  # <--- NEW: Capture Comment ID for Deep Linking
                    "DayOfWeek": date_obj.strftime('%A'),
                    "Hour": date_obj.hour
                })
        except: continue
        
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(all_comments)

# --- UI Layout ---

with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    
    url_input = st.text_input("Channel/Video URL", "https://www.youtube.com/@ThePrimeagen")
    
    c1, c2 = st.columns(2)
    with c1: sort_mode = st.selectbox("Sort By", ["Latest", "Popular"])
    with c2: video_limit = st.number_input("Videos", 1, 20, 3)
    
    comment_limit = st.slider("Comments/Video", 50, 500, 100)
    
    st.markdown("### 🚨 Watchlist Keywords")
    alert_keywords = st.text_area("Separated by comma", "scam, fraud, refund, broken, fail, hate, worst").split(',')
    alert_keywords = [k.strip().lower() for k in alert_keywords if k.strip()]
    
    st.markdown("---")
    run_btn = st.button("🚀 Generate Report", type="primary")

st.title("🧠 Social Media Intelligence Report")
st.markdown("Advanced analytics for community sentiment, emerging trends, and automated response drafting.")

# --- Main Logic ---
if run_btn:
    with st.spinner('Gathering Intelligence...'):
        df = scrape_data(url_input, video_limit, comment_limit, sort_mode)
        st.session_state['data'] = df

if 'data' in st.session_state and not st.session_state['data'].empty:
    df = st.session_state['data']
    
    # --- KPI HEADER ---
    col1, col2, col3, col4 = st.columns(4)
    avg_score = df['Sentiment Score'].mean()
    risky_comments = df[df['Comment'].str.contains('|'.join(alert_keywords), case=False, na=False)]
    
    col1.metric("Total Volume", f"{len(df)} Comments")
    col2.metric("Net Sentiment", f"{avg_score:.2f}", delta="Positive" if avg_score > 0 else "Negative")
    col3.metric("Total Engagement", f"{df['Likes'].sum():,}")
    col4.metric("🚨 Critical Alerts", len(risky_comments), delta="-Action Required" if len(risky_comments) > 0 else "Clean", delta_color="inverse")

    # --- TABS ---
    t_actions, t_insight, t_visual, t_data = st.tabs(["⚡ Response Center", "📊 Analytics", "🎨 Visualization", "💾 Raw Data"])

    # --- TAB 1: ACTION CENTER ---
    with t_actions:
        st.subheader("🤖 AI Response Assistant")
        st.markdown("Select a comment to generate a context-aware professional response.")
        
        # Filter Selection
        risk_filter = st.radio("Filter Feed:", ["All Comments", "Negative Only", "Watchlist Keywords"], horizontal=True)
        
        if risk_filter == "Negative Only":
            display_df = df[df['Sentiment'] == 'Negative']
        elif risk_filter == "Watchlist Keywords":
            display_df = risky_comments
        else:
            display_df = df
            
        if not display_df.empty:
            # We need the original index to access the full dataframe data safely
            selected_idx = st.selectbox(
                "Select a Comment:", 
                display_df.index, 
                format_func=lambda x: f"{display_df.loc[x, 'Author']}: {display_df.loc[x, 'Comment'][:60]}..."
            )
            
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                comment_data = display_df.loc[selected_idx]
                st.info(f"**User said:**\n\n\"{comment_data['Comment']}\"\n\n*Likes: {comment_data['Likes']} | Sentiment: {comment_data['Sentiment']}*")
                
                # --- NEW BUTTONS LAYOUT ---
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("✨ Draft Response", use_container_width=True):
                        with st.spinner("Analyzing context..."):
                            reply = get_ai_response(comment_data['Comment'], api_key)
                            st.session_state['generated_reply'] = reply
                
                with btn_col2:
                    # Construct Deep Link to the specific comment
                    # Note: scraping sometimes misses the 'cid'. We fallback to video URL if missing.
                    if 'cid' in comment_data and comment_data['cid']:
                        comment_url = f"{comment_data['Video URL']}&lc={comment_data['cid']}"
                    else:
                        comment_url = comment_data['Video URL']
                        
                    st.link_button("↗️ Open in YouTube", comment_url, use_container_width=True)
            
            with col_b:
                st.markdown("**Suggested Reply:**")
                if 'generated_reply' in st.session_state:
                    st.text_area("Copy to Clipboard:", st.session_state['generated_reply'], height=150)
                else:
                    st.markdown("*Click 'Draft Response' to generate a reply.*")
                    st.markdown("*Click 'Open in YouTube' to go directly to this comment.*")
        else:
            st.success("No comments found matching this filter.")

    # --- TAB 2: ANALYTICS ---
    with t_insight:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Sentiment Timeline")
            trend = df.set_index('Date').resample('D')['Sentiment Score'].mean().reset_index()
            fig = px.line(trend, x='Date', y='Sentiment Score', markers=True, title="Brand Health Over Time")
            fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1, line_width=0)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Engagement Matrix")
            vid_stats = df.groupby('Video Title').agg({'Sentiment Score': 'mean', 'Likes': 'sum', 'Comment': 'count'}).reset_index()
            fig_bub = px.scatter(vid_stats, x='Sentiment Score', y='Likes', size='Comment', hover_name='Video Title', 
                                 color='Sentiment Score', color_continuous_scale='RdBu', title="Engagement vs Sentiment")
            st.plotly_chart(fig_bub, use_container_width=True)

    # --- TAB 3: VISUALS ---
    with t_visual:
        st.subheader("Topic Modeling")
        col_wc1, col_wc2 = st.columns([2, 1])
        
        with col_wc1:
            st.markdown("**Context Word Cloud**")
            if not df.empty:
                text = " ".join(df['Comment'].astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
                
        with col_wc2:
            st.markdown("**Key Phrases**")
            phrases = extract_topics(df['Comment'])
            if not phrases.empty:
                st.dataframe(phrases, use_container_width=True, height=350)

    # --- TAB 4: DATA ---
    with t_data:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Export Report CSV", df.to_csv(index=False).encode('utf-8'), "social_intelligence_report.csv", "text/csv")

elif run_btn:
    st.warning("No data found. Please check the URL.")
else:
    st.info("👈 Enter a URL and settings in the sidebar to begin.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    <p>Made with ❤️ by <b class="highlight">Saif Ali</b></p>
    <p>
        <a href="https://www.sudobotz.com" target="_blank">🌐 sudobotz.com</a> • 
        <a href="mailto:contact@sudobotz.com">📧 contact@sudobotz.com</a> • 
        <a href="https://fiverr.com/saifalimz" target="_blank">💼 Hire me on Fiverr</a>
    </p>
</div>
""", unsafe_allow_html=True)

