import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import requests
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities

import warnings

warnings.filterwarnings("ignore")

@st.cache_data
def get_data():
    df = pd.read_csv("IMDB_TOP_1000_RATED_DESCENDING.csv", encoding='latin1')
    return df


#-----------Web page setting-------------------#
page_title = "Movie Recommendation App"
page_icon = ":robot"
#approve_icon = ":check_mark_button:"
#not_approve_icon = ":prohibited:"
layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)


# Set up Menu/Navigation Bar
selected = option_menu(
    menu_title = "MovieRecommender",
    options = ['Home', 'Explore', 'Get Recommendation', 'Contact'],
    icons = ["house-fill", "book-half", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

# Set `Home` page
if selected == "Home":
    st.title("Welcome!")
    st.write("MovieRecommender is a web app that uses NLP algorithm to recommend similar movies to watch based on the movie synopsis.")
    st.markdown("""The data is scraped from [IMDb Top 1000 (Sorted by User rating Descending)](https://www.imdb.com/search/title/?count=100&groups=top_1000&sort=user_rating).""") 
    st.write("This is then processed, analyzed and used to train a TFIDF model and the Gensim's Similarities packege is used to compute the similarity index.")
    st.write("Users can explore the data as well as imput their favourite movie to see which ones have similar synopsis to the selected one.")


# Set `Explore` page
if selected == "Explore":
    data = get_data()
    
    st.write("""### Explore the Movie Data Data""")

    # Set chart options
    chart_opt = [
        "Trend of Ratings", 
        "Most Rated Movies by Year", 
        "To 10 Rated Movies of all Time"
        ]
    
    # Plotting analysis
    chart = st.selectbox("Select analysis:", chart_opt)

    if chart == "Trend of Ratings":
        fig1 = plt.figure()
        sns.lineplot(x='YEAR', y='RATING', data=data, ci=None)
        plt.title('Trend of Ratings Over the Years')
        st.pyplot(fig1)

    if chart == "Most Rated Movies by Year":
        trend = data.groupby('YEAR')['TITLE'].count()

        fig2 = plt.figure()
        sns.lineplot(x=trend.index, y=trend.values, data=trend, ci=None)
        plt.title('Trend of Ratings Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Count of Movies')
        st.pyplot(fig2)


    if chart == "To 10 Rated Movies of all Time":
        mask = data[['TITLE', 'RATING']].head(10)

        fig3 = plt.figure()
        sns.barplot(x='RATING', y='TITLE', data=mask)
        plt.title('Top 10 Most Rated Movies')
        plt.xlabel('Rating')
        plt.ylabel('Movie Title')
        st.pyplot(fig3)

# Set `Recommendation` page
if selected == 'Get Recommendation':
    data = get_data()
    # Load NLTK stopwords from the saved file
    with open('stopwords.txt', 'r') as f:
        stop_words = set(f.read().splitlines())
    
    # Transform the text into tokens 
    data['Synopsis_processed'] = data['SYNOPSIS'].str.split()

    # Remove stop words
    data['Synopsis_processed']= data['Synopsis_processed'].apply(lambda x: [word for word in x if word not in stop_words])

    # Create stemmer object
    snowball_stemmer = SnowballStemmer('english')

    # Define stemming function
    def apply_stemming(tokens):
        return [snowball_stemmer.stem(token) for token in tokens]
    
    # Apply stemming function
    data['Synopsis_processed'] = data['Synopsis_processed'].apply(apply_stemming)

    # Get list of tokenized texts
    tokenized_texts = data['Synopsis_processed'].tolist()

    # Map words to ids
    word_dict = Dictionary(tokenized_texts)

    # BoW representation
    corpus_bow = [word_dict.doc2bow(text) for text in tokenized_texts]

    # Create TFIDF model
    tfidf_model = TfidfModel(corpus_bow)

    # Compute the similarity matrix (pairwise distance between all texts)
    sims = similarities.MatrixSimilarity(tfidf_model[corpus_bow])

    # Transform the resulting list into a dataframe
    sim_df = pd.DataFrame(list(sims))

    # Add the titles of the books as columns and index of the dataframe
    sim_df.columns = data['TITLE']
    sim_df.index = data['TITLE']


    top_movies = data[['TITLE', 'RATING']].sort_values(by='RATING', ascending=False)

    col_name = st.selectbox("Select Movie Title", (top_movies['TITLE'].values.tolist()))

    if col_name:
        # Get the most similar books based on the selected book
        v = sim_df[col_name].sort_values(ascending=False).iloc[1:].head(10)

        # Sort by ascending scores
        v = v.sort_values(ascending=True)
        fig = px.bar(
            v, 
            x=v.values, 
            y=v.index, 
            orientation='h',  # Horizontal orientation
            labels={'x': 'Similarity Index', 'y': 'Movie Title'},
            title=f"Top 10 Most Similar Movies to '{col_name}'"
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)

# Set `Contact` page
if selected == "Contact":
    # Contact Web App
    st.write("""### Get in touch""")
    st.markdown("""Email: [Link](mailto:gamahrichard5@gmail.com).""")

    st.markdown("""GitHub: [Link](https://github.com/SirGamah/).""")

    st.markdown("""WhatsApp: [Link](https://wa.me/233542124371).""")
    
