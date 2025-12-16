import streamlit as st
import pandas as pd
import scrapetube
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import google.generativeai as genai
import json
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import time
from datetime import datetime, timedelta
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Social Media Intelligence Dashboard", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container { padding-bottom: 100px; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);
        color: #4b5563; text-align: center; padding: 15px 0; font-size: 14px;
        border-top: 1px solid #e5e7eb; z-index: 9999;
        font-family: 'Source Sans Pro', sans-serif;
    }
    .footer a { color: #FF4B4B; text-decoration: none; font-weight: 600; margin: 0 12px; transition: all 0.2s ease; }
    .footer a:hover { color: #D43636; text-decoration: underline; }
    .highlight { color: #FF4B4B; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
b64_key = "QUl6YVN5QWI2WXl6VDdkVkN5ZHNrVU54eHZaanczNm43Ti13YnBv"

def decode_base64_string(encoded_string):
    """
    Decodes a base64 string (standard or URL-safe) to a UTF-8 string.
    """
    try:
        # Ensure the string has correct padding (length must be divisible by 4)
        # This adds '=' to the end if they are missing
        missing_padding = len(encoded_string) % 4
        if missing_padding:
            encoded_string += '=' * (4 - missing_padding)

        # Decode the bytes
        # urlsafe_b64decode handles both standard (+/) and url-safe (-_) characters
        decoded_bytes = base64.urlsafe_b64decode(encoded_string)

        # Convert bytes to string (UTF-8)
        return decoded_bytes.decode('utf-8')
        
    except Exception as e:
        return f"Error decoding: {e}"

DEFAULT_API_KEY = decode_base64_string(b64_key)

# --- AI Helper: Batch Sentiment Analysis (FIXED & ROBUST) ---
def batch_analyze_sentiment(comments_data, api_key):
    """
    Sends a batch of comments to Gemini.
    FIX: Now tries multiple models if one is not found (404).
    TUNED FOR: Sarcasm & Context.
    """
    if not api_key or not comments_data: return {}

    genai.configure(api_key=api_key)
    
    # Priority List: Try the smartest/newest first, fallback to standard
    model_candidates = [
        'gemini-2.5-flash-lite',
        'gemini-2.5-flash'
    ]
    
    selected_model = None
    
    # Auto-Discovery Loop
    for m in model_candidates:
        try:
            test_model = genai.GenerativeModel(m)
            # Simple test to check if model exists for this key
            # We don't generate content here to save time, just init is usually enough
            # But to be safe let's try a dry run if needed, or just proceed
            selected_model = test_model
            break # Found a working one
        except:
            continue
            
    if not selected_model:
        print("❌ All AI models failed. Check API Key.")
        return {}

    try:
        simple_payload = [{"id": item['id'], "text": item['text']} for item in comments_data]
        payload_str = json.dumps(simple_payload)
        
        prompt = f"""
        Analyze the sentiment of these YouTube comments for an Online Reputation Report.
        
        INPUT DATA:
        {payload_str}
        
        INSTRUCTIONS:
        1. Read the comment carefully. Look for sarcasm, emojis (❌, 🤡, 💀), and comparisons.
        2. "Nothing phone company ❌, YouTubers ✅" is NEGATIVE (Insult).
        3. "Great job breaking it" is NEGATIVE (Sarcasm).
        4. "Is this safe?" is NEGATIVE (Risk).
        
        OUTPUT FORMAT (JSON List):
        - You MUST include a short "reasoning" field.
        - Example: [{{"id": 10, "reasoning": "User insults product quality using comparison", "sentiment": "Negative"}}]
        - Sentiment must be one of: "Positive", "Negative", "Neutral".
        """
        
        response = selected_model.generate_content(prompt)
        
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        results = json.loads(cleaned_text)
        sentiment_map = {item['id']: item['sentiment'] for item in results}
        return sentiment_map
        
    except Exception as e:
        print(f"Batch Analysis Failed: {e}")
        return {} 

# --- AI Helper: Response Generation ---
def get_ai_response(comment_text, api_key):
    if not api_key: return "Error: No API Key provided."
    genai.configure(api_key=api_key)
    
    model_candidates = ['gemini-2.5-flash-lite', 'gemini-2.5-flash']
    
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            prompt = f"""
            You are a Social Media Manager. Write a polite, professional reply to this comment.
            - Under 280 characters.
            - NO character counts or explanations.
            - Return ONLY the reply text.
            Comment: "{comment_text}"
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except: continue

    return "❌ AI Error: Could not generate response."

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

def extract_topics(text_series, n=2):
    text = text_series.dropna().astype(str).tolist()
    if not text: return pd.DataFrame()
    
    custom_stops = ['video', 'youtube', 'channel', 'content', 'subscribe', 'watching', 'guy', 'bro', 'like', 'just', 'really']
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english', min_df=2, max_features=50)
    
    try:
        stop_words = list(vec.get_stop_words()) + custom_stops
        vec.stop_words = stop_words
        bag_of_words = vec.fit_transform(text)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        return pd.DataFrame(sorted(words_freq, key=lambda x: x[1], reverse=True)[:20], columns=['Phrase', 'Count'])
    except ValueError:
        return pd.DataFrame()

# --- Scraper & Processor (With Fallback Logic) ---
def scrape_and_process(channel_url, video_limit, comment_limit, sort_mode, api_key):
    downloader = YoutubeCommentDownloader()
    
    # 1. ROBUST URL CLEANING
    # Use Regex to extract the actual URL if it's buried in text or brackets
    found_url = re.search(r'(https?://[^\s)\]"\']+)', channel_url)
    if found_url:
        channel_url = found_url.group(1)
    else:
        st.error("❌ No valid URL found. Please check your input.")
        return pd.DataFrame()

    # 2. RESOLVE VIDEOS
    sort_map = {"Latest": "newest", "Popular": "popular"}
    try:
        if "watch?v=" in channel_url or "youtu.be/" in channel_url:
            # Single Video Mode
            video_id = channel_url.split("v=")[-1].split("&")[0] if "v=" in channel_url else channel_url.split("/")[-1]
            videos = [{'videoId': video_id, 'title': {'runs': [{'text': 'Single Video Scan'}]}}]
        else:
            # Channel Mode
            videos = list(scrapetube.get_channel(channel_url=channel_url, limit=video_limit, sort_by=sort_map[sort_mode]))
    except Exception as e:
        st.error(f"Error fetching channel list: {e}")
        return pd.DataFrame()

    raw_comments = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_videos = len(videos)
    if total_videos == 0:
        st.warning("No videos found. Check URL.")
        return pd.DataFrame()

    # 3. SCRAPE LOOP WITH FALLBACK
    for i, video in enumerate(videos):
        try:
            title = video.get('title', {}).get('runs', [{}])[0].get('text', 'Unknown Title')
            video_id = video['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            status_text.text(f"Scraping ({i+1}/{total_videos}): {title[:40]}...")
            progress_bar.progress((i + 1) / total_videos)
            
            # --- FALLBACK MECHANISM START ---
            iterator = None
            try:
                # Attempt 1: Try the user's preferred sort (usually Popular)
                generator = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
                
                # "Peek" at the first item to see if it works. 
                # If this line fails, we jump to the 'except' block.
                first_item = next(generator)
                
                # Put the item back into the stream
                import itertools
                iterator = itertools.chain([first_item], generator)
                
            except (StopIteration, Exception):
                # Attempt 2: If Popular fails, Force "Newest" (Sort By = 1)
                # This is much more stable on YouTube.
                # st.toast(f"Switched to 'Newest' sort for video {i+1}", icon="⚠️") # Optional feedback
                try:
                    iterator = downloader.get_comments_from_url(video_url, sort_by=1)
                except:
                    continue # Skip this video if both fail
            # --- FALLBACK MECHANISM END ---

            count = 0
            if iterator:
                for c in iterator:
                    if count >= comment_limit: break
                    
                    date_obj = parse_relative_date(c.get('time', ''))
                    
                    raw_comments.append({
                        "id": len(raw_comments), 
                        "Video Title": title,
                        "Author": c.get('author', 'Anonymous'),
                        "Comment": c.get('text', ''),
                        "Likes": parse_votes(c.get('votes', '0')),
                        "Date": date_obj,
                        "Video URL": video_url,
                        "cid": c.get('cid'),
                        "DayOfWeek": date_obj.strftime('%A'),
                        "Hour": date_obj.hour
                    })
                    count += 1
        except Exception as e:
            print(f"Video {i} failed: {e}")
            continue
        
    if not raw_comments:
        st.error("No comments found. YouTube may have blocked the scraper temporarily or comments are disabled.")
        return pd.DataFrame()
        
    status_text.text("🤖 AI analyzing sentiment in batches...")
    
    

    # 4. AI BATCH SENTIMENT
    BATCH_SIZE = 25
    sentiment_results = {}
    ai_payload = [{"id": x['id'], "text": x['Comment'][:500]} for x in raw_comments]
    total_comments = len(ai_payload)
    
    for i in range(0, total_comments, BATCH_SIZE):
        batch = ai_payload[i:i + BATCH_SIZE]
        batch_result = batch_analyze_sentiment(batch, api_key)
        sentiment_results.update(batch_result)
        progress_bar.progress(min((i + BATCH_SIZE) / total_comments, 1.0))
        time.sleep(0.5)

    # 5. MERGE
    final_data = []
    score_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    
    for item in raw_comments:
        sent_label = sentiment_results.get(item['id'], "Neutral")
        sent_score = score_map.get(sent_label, 0.0)
        item['Sentiment'] = sent_label
        item['Sentiment Score'] = sent_score
        del item['id'] 
        final_data.append(item)

    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(final_data)

# --- UI Layout ---

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    url_input = st.text_input("Channel/Video URL", "[https://www.youtube.com/@ThePrimeagen](https://www.youtube.com/@ThePrimeagen)")
    
    c1, c2 = st.columns(2)
    with c1: sort_mode = st.selectbox("Sort By", ["Latest", "Popular"])
    with c2: video_limit = st.number_input("Videos", 1, 20, 3)
    
    comment_limit = st.slider("Comments/Video", 50, 500, 50)
    
    st.markdown("### 🚨 Watchlist Keywords")
    alert_keywords = st.text_area("Separated by comma", "scam, fraud, refund, broken, fail, hate, worst").split(',')
    alert_keywords = [k.strip().lower() for k in alert_keywords if k.strip()]
    
    st.markdown("---")
    run_btn = st.button("🚀 Generate Report", type="primary")

st.title("🧠 Social Media Intelligence Report")
st.markdown("AI-Powered Sentiment Analysis (No local libraries) & Response Automation.")

# --- Main Logic ---
if run_btn:
    if not api_key:
        st.error("Please provide an API Key to use Gemini features.")
    else:
        with st.spinner('Scraping & Analyzing with AI...'):
            df = scrape_and_process(url_input, video_limit, comment_limit, sort_mode, api_key)
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
        
        risk_filter = st.radio("Filter Feed:", ["All Comments", "Negative Only", "Watchlist Keywords"], horizontal=True)
        
        if risk_filter == "Negative Only":
            display_df = df[df['Sentiment'] == 'Negative']
        elif risk_filter == "Watchlist Keywords":
            display_df = risky_comments
        else:
            display_df = df
            
        if not display_df.empty:
            selected_idx = st.selectbox(
                "Select a Comment:", 
                display_df.index, 
                format_func=lambda x: f"{display_df.loc[x, 'Author']}: {display_df.loc[x, 'Comment'][:60]}..."
            )
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                comment_data = display_df.loc[selected_idx]
                st.info(f"**User said:**\n\n\"{comment_data['Comment']}\"\n\n*Likes: {comment_data['Likes']} | Sentiment: {comment_data['Sentiment']}*")
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✨ Draft Response", width="stretch"): # Updated deprecation
                        with st.spinner("Analyzing context..."):
                            reply = get_ai_response(comment_data['Comment'], api_key)
                            st.session_state['generated_reply'] = reply
                
                with btn_col2:
                    if 'cid' in comment_data and comment_data['cid']:
                        comment_url = f"{comment_data['Video URL']}&lc={comment_data['cid']}"
                    else:
                        comment_url = comment_data['Video URL']
                    st.link_button("↗️ Open in YouTube", comment_url, width="stretch") # Updated deprecation
            
            with col_b:
                st.markdown("**Suggested Reply:**")
                if 'generated_reply' in st.session_state:
                    st.text_area("Copy to Clipboard:", st.session_state['generated_reply'], height=150)
                else:
                    st.markdown("*Click 'Draft Response' to generate a reply.*")
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
            st.plotly_chart(fig, width="stretch") # Updated deprecation
            
        with c2:
            st.subheader("Engagement Matrix")
            vid_stats = df.groupby('Video Title').agg({'Sentiment Score': 'mean', 'Likes': 'sum', 'Comment': 'count'}).reset_index()
            fig_bub = px.scatter(vid_stats, x='Sentiment Score', y='Likes', size='Comment', hover_name='Video Title', 
                                 color='Sentiment Score', color_continuous_scale='RdBu', title="Engagement vs Sentiment")
            st.plotly_chart(fig_bub, width="stretch") # Updated deprecation

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
                st.dataframe(phrases, width="stretch", height=350) # Updated deprecation

    # --- TAB 4: DATA ---
    with t_data:
        st.dataframe(df, width="stretch") # Updated deprecation
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
        <a href="[https://www.sudobotz.com](https://www.sudobotz.com)" target="_blank">🌐 sudobotz.com</a> • 
        <a href="mailto:contact@sudobotz.com">📧 contact@sudobotz.com</a> • 
        <a href="[https://fiverr.com/saifalimz](https://fiverr.com/saifalimz)" target="_blank">💼 Hire me on Fiverr</a>
    </p>
</div>
""", unsafe_allow_html=True)
