import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(layout="wide", page_title="Farsight Media Social Media Analyzer", page_icon="📊")

# Enhanced custom CSS
st.markdown("""
<style>
    body {
        color: #333;
        font-family: 'Helvetica', 'Arial', sans-serif;
    }
    .reportview-container {
        background: linear-gradient(120deg, #f6f9fc 0%, #e9f1f9 100%);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    .Widget>label {
        color: #1F3B73;
        font-weight: bold;
        font-size: 1.1em;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1F3B73;
        border-radius: 5px;
        padding: 0.5em 1em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF7A00;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        border-color: #1F3B73;
        padding: 0.5em;
    }
    h1 {
        color: #1F3B73;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    h2 {
        color: #FF7A00;
        font-size: 1.8em;
        margin-top: 1em;
    }
    .stProgress > div > div > div > div {
        background-color: #FF7A00;
    }
    .plot-container {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 1em;
        background: white;
    }
    .card {
        border-radius: 10px;
        padding: 1.5em;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1em;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #1F3B73;
    }
    .metric-label {
        font-size: 1.2em;
        color: #FF7A00;
    }
</style>
""", unsafe_allow_html=True)

EXCEL_FILE = 'Farsight.xlsx'
MODEL_PATH = './fine_tuned_model'
SENTIMENT_LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Load the model and tokenizer with error handling for safetensors
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Attempt to load model using safetensors
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model with safetensors: {e}")
        
        try:
            # Fallback to loading the model without safetensors
            model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
            tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading model without safetensors: {e}")
            return None, None

# Load data with error handling
@st.cache
def load_data():
    try:
        df = pd.read_excel(EXCEL_FILE)
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'Farsight.xlsx' is in the correct location.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Sentiment analysis function with error handling
def analyze_sentiment_bert(text, model, tokenizer):
    if not model or not tokenizer:
        st.warning("Model or tokenizer not loaded properly. Sentiment analysis cannot be performed.")
        return "N/A"
    
    try:
        with st.spinner('Analyzing sentiment...'):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            sentiment_label = torch.argmax(probs).item()
        return SENTIMENT_LABELS.get(sentiment_label, "N/A")
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return "Error"

# Filter dataframe based on keyword
def filter_data_by_keyword(df, keyword):
    if keyword:
        return df[df['Content'].str.contains(keyword, case=False, na=False)]
    return df

# Filter dataframe based on sidebar filters
def filter_dataframe(df, category, source, tonality, theme, date_range):
    try:
        filtered_df = df.copy()
        if category:
            filtered_df = filtered_df[filtered_df['Category'].isin(category)]
        if source:
            filtered_df = filtered_df[filtered_df['Source'].isin(source)]
        if tonality:
            filtered_df = filtered_df[filtered_df['Tonality'].isin(tonality)]
        if theme:
            filtered_df = filtered_df[filtered_df['Theme'].isin(theme)]
        if date_range:
            filtered_df = filtered_df[
                (filtered_df['Date'] >= pd.to_datetime(date_range[0])) & 
                (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
            ]
        return filtered_df
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return df

# Home page
def home_page():
    st.title('Farsight Social Listening and Classification System')

    # Load the data and model
    df = load_data()
    model, tokenizer = load_model()

    if df.empty:
        st.warning("No data available to display.")
        return

    # Calculate date range from the data
    min_date = df['Date'].min()
    max_date = df['Date'].max()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.button('🖥️ PowerBI Dashboard', on_click=set_page, args=('dashboard',), key='dashboard_button')

    with col2:
        search_term = st.text_input("Enter keyword or Topic for search:", placeholder="Type here to search...")

    # Sidebar filters with error handling
    with st.sidebar:
        try:
            st.sidebar.image("https://media.licdn.com/dms/image/v2/D4D0BAQFk-Wh7z9QcoA/company-logo_200_200/company-logo_200_200/0/1685437983213/prescott_data_logo?e=2147483647&v=beta&t=w9MP41RnNmTWMvMwS_HqcbUeCAegtj6zuB4VaSFhH6M", width=160)
            st.sidebar.title("🔍 Filters")
            category = st.sidebar.multiselect('📁 Category', df['Category'].unique())

            # Update Source filter based on selected Category
            if category:
                filtered_sources = df[df['Category'].isin(category)]['Source'].unique()
                source = st.sidebar.multiselect('📰 Source', filtered_sources)
            else:
                source = st.sidebar.multiselect('📰 Source', df['Source'].unique())

            tonality = st.sidebar.multiselect('Tonality', df['Tonality'].unique())
            theme = st.sidebar.multiselect('Theme', df['Theme'].unique())

            # Use min and max dates from the dataset for the date range filter
            date_range = st.sidebar.date_input('📅 Date Range', [min_date, max_date], min_value=min_date, max_value=max_date)
        except Exception as e:
            st.error(f"Error setting filters: {e}")

    # Filter data
    filtered_df = filter_data_by_keyword(df, search_term)
    filtered_df = filter_dataframe(filtered_df, category, source, tonality, theme, date_range)

    # Display key metrics based on the filtered data or overall data
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    try:
        with col1:
            st.markdown('<div class="card">'
                        f'<div class="metric-value">{len(filtered_df):,}</div>'
                        '<div class="metric-label">Total Posts</div>'
                        '</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card">'
                        f'<div class="metric-value">{filtered_df["Category"].nunique()}</div>'
                        '<div class="metric-label">Categories</div>'
                        '</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card">'
                        f'<div class="metric-value">{filtered_df["Theme"].nunique()}</div>'
                        '<div class="metric-label">Themes</div>'
                        '</div>', unsafe_allow_html=True)
        with col4:
            avg_sentiment = filtered_df['Tonality'].value_counts(normalize=True).get('Positive', 0)
            st.markdown('<div class="card">'
                        f'<div class="metric-value">{avg_sentiment:.2%}</div>'
                        '<div class="metric-label">Positive Sentiment</div>'
                        '</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying key metrics: {e}")

    # Display search results and data
    if search_term:
        if filtered_df.empty:
            st.write("No results found.")
        else:
            display_search_results(filtered_df, model, tokenizer)
            st.subheader("Word Cloud")
            create_wordcloud(' '.join(filtered_df['Content']))
    else:
        st.subheader("Data Sample")
        try:
            display_df = filtered_df.copy()
            for col in display_df.select_dtypes(include=['datetime64']).columns:
                display_df[col] = display_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_df)
        except Exception as e:
            st.error(f"Error displaying data sample: {e}")

    # Dataset Overview
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Distribution")
            fig1 = px.pie(filtered_df, names='Category', title='Posts by Category',
                          color_discrete_sequence=['#1F3B73', '#A2B9E5', '#FF7A00'])
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(margin=dict(t=50, b=50, l=20, r=20))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(filtered_df['Tonality'].value_counts(), title='Tonality Distribution',
                          color_discrete_sequence=['#1F3B73', '#A2B9E5', '#FF7A00'])
            fig2.update_layout(xaxis_title="Tonality", yaxis_title="Count", margin=dict(t=50, b=50, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying dataset overview: {e}")

    # Sentiment Analysis
    st.header("Real-Time Sentiment Analysis")
    user_input = st.text_area("Enter text for analysis:", placeholder="Type or paste your text here...")
    if user_input:
        sentiment = analyze_sentiment_bert(user_input, model, tokenizer)
        sentiment_color = {'Positive': '#FF7A00', 'Neutral': '#A2B9E5', 'Negative': '#1F3B73'}
        color = sentiment_color.get(sentiment, '#000000')  # Default to black if sentiment is not recognized
        st.markdown(f"Predicted Sentiment: <span style='color:{color};font-weight:bold;font-size:24px;'>{sentiment}</span>", unsafe_allow_html=True)
        try:
            confidence = torch.softmax(model(**tokenizer(user_input, return_tensors='pt', truncation=True, padding=True, max_length=128)).logits, dim=1)[0]
            confidence_df = pd.DataFrame({'Sentiment': list(SENTIMENT_LABELS.values()), 'Confidence': confidence.tolist()})
            confidence_chart = alt.Chart(confidence_df).mark_bar().encode(
                x='Sentiment',
                y='Confidence',
                color=alt.Color('Sentiment', scale=alt.Scale(domain=list(SENTIMENT_LABELS.values()), range=['#1F3B73', '#A2B9E5', '#FF7A00'])),
                tooltip=['Sentiment', alt.Tooltip('Confidence', format='.2%')]
            ).properties(title='Sentiment Confidence').interactive()
            st.altair_chart(confidence_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying sentiment confidence: {e}")

# Dashboard page
def dashboard_page():
    st.title("🖥️ PowerBi Dashboard")
    st.button('🏠 Back to Home', on_click=set_page, args=('home',), key='home_button')
    
    try:
        st.markdown("""
        <div style="padding:20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #1F3B73;">Welcome to the PowerBi Dashboard</h3>
            <p>This dashboard provides insights into the social media data. Exploring trends, sentiment analysis, and key metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.iframe(
            "https://app.powerbi.com/view?r=eyJrIjoiNjBiNmNkNzAtNjkyMi00Y2FhLTgzMmItMDk3ZmEyODk1ZTYwIiwidCI6IjYyZmJhNjA1LThhMjktNDZhYS1hZDU0LTcyZjgwNmMwZWY1YSJ9",
            width=1140, 
            height=541
        )
    except Exception as e:
        st.error(f"Error loading PowerBi Dashboard: {e}")

# Page navigation
def set_page(page_name):
    st.session_state.page = page_name

# Main app
def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "dashboard":
        dashboard_page()

if __name__ == "__main__":
    main()
