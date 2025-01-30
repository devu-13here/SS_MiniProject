import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


st.title('Sentiment Analysis of Tweets about US Airlines')
st.sidebar.title("Sentiment Analysis of Tweets about US Airlines")

st.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets 🐦")

# Correct the file path (use raw string or forward slashes)
DATA_URL = r'C:/Users/patel/OneDrive/Desktop/newosfolder/NNNEW/SEMESTER 6/Computer Graphics/Project/Tweets.csv'

# Updated caching function
@st.cache_data(persist=True)
def load_data(DATA_URL):
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    
    # Check if 'tweet_coord' is a string before applying eval
    def safe_eval(coord):
        if isinstance(coord, str):
            return eval(coord)
        return None

    # Apply safe_eval to 'tweet_coord' column
    data['tweet_coord'] = data['tweet_coord'].apply(safe_eval)

    # Create 'latitude' and 'longitude' columns by splitting 'tweet_coord'
    data['latitude'] = data['tweet_coord'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 2 else None)
    data['longitude'] = data['tweet_coord'].apply(lambda x: x[1] if isinstance(x, list) and len(x) == 2 else None)
    
    return data

# Load data
data = load_data(DATA_URL)

# Show random tweet
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0, 0])

# Number of tweets by sentiment
st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Histogram', 'Pie chart'], key='2')  # Changed key to '2'
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Histogram':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

# When and where are users tweeting from?
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour of day", 0 ,23)
modified_data = data[data['tweet_created'].dt.hour == hour]

if not st.sidebar.checkbox("Close", True , key='3'):  # Changed key to '3'
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    
    # Drop rows without valid latitude and longitude
    modified_data = modified_data.dropna(subset=['latitude', 'longitude'])
    
    # Ensure data has valid latitude and longitude before using st.map()
    if not modified_data.empty:
        st.map(modified_data[['latitude', 'longitude']])
    else:
        st.write("No location data available for the selected time.")
    
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)

st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice = st.sidebar.multiselect('Pick Airline', ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America') , key='0')

if len(choice) > 0:
    choice_data = data[data['airline'] == choice[0]]
    fig_choice = px.histogram(choice_data, x='airline' ,y='airline_sentiment', histfunc='count',color="airline_sentiment" , facet_col='airline_sentiment' , labels ={'airline_sentiment' : "tweets"}, height=600, width=800)
    st.plotly_chart(fig_choice)
    
    
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Close", True, key='4'):
    st.subheader(f"Word cloud for {word_sentiment} sentiment")
    df = data[data['airline_sentiment'] == word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

