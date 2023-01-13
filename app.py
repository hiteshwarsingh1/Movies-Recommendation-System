import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# from tmdbv3api import TMDb
# tmdb = TMDb()
# tmdb.api_key = 'YOUR_API_KEY'

# from tmdbv3api import Movie

# load the nlp model and tfidf vectorizer from disk
filename = 'sentiment.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def get_similarity():
    data = pd.read_csv('q_movies.csv')
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    data['overview'] = data['overview'].fillna('')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return data, cosine_sim

# def get_path(imdb_id):
#     url = 'https://www.imdb.com/title/{}/mediaviewer/rm1973718272/?ref_=tt_ov_i'.format(imdb_id)
#     HEADERS ={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}
#     resp = requests.get(url, headers=HEADERS)
#     content = bs.BeautifulSoup(resp.content, 'lxml')

#     casts = content.select('.sc-bfec09a1-1')
#     # print(casts)
#     cast_names =[]
#     for cast in casts:
#         cast_names.append(cast.get_text().strip())

#     return cast_names

def rcmd(m):
    data, cosine_sim= get_similarity()

    if m not in data['title'].unique():
        print("ddddddddd")
        return ('Sorry! The movie your searched is not in our database. Please check the spelling or try with some other movies'), None
    else:
        try:
            idx = data.loc[data['title'] == m, 'Idx'].values[0]
        except:
            return "Movie not found", None
        # idx = data.loc[data['title'] == m, 'Idx'].values[0]
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        imdb_id = data['imdb_id'].loc[data['title']==m].values[0]

        # Return the top 10 most similar movies
        return data['title'].iloc[movie_indices].values, imdb_id

# def ListOfGenres(genre_json):
#     if genre_json:
#         genres = []
#         genre_str = ", " 
#         for i in range(0,len(genre_json)):
#             genres.append(genre_json[i]['name'])
#         return genre_str.join(genres)

# def date_convert(s):
#     MONTHS = ['January', 'February', 'Match', 'April', 'May', 'June',
#     'July', 'August', 'September', 'October', 'November', 'December']
#     y = s[:4]
#     m = int(s[5:-3])
#     d = s[8:]
#     month_name = MONTHS[m-1]

#     result= month_name + ' ' + d + ' '  + y
#     return result

# def MinsToHours(duration):
#     if duration%60==0:
#         return "{:.0f} hours".format(duration/60)
#     else:
#         return "{:.0f} hours {} minutes".format(duration/60,duration%60)

def get_suggestions():
    data = pd.read_csv('movies_metadata.csv')
    return list(data['title'].str.capitalize())


app = Flask(__name__)

@app.route("/")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions = suggestions)


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie') # get movie name from the URL
    movie = movie.title()
    r, imdb_id = rcmd(movie)
    if type(r)==type('string') or imdb_id==None: # no such movie found in the database
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce,'lxml')
        soup_result = soup.find_all("div",{"class":"text show-more__control"})

        reviews_list = [] # list of reviews
        reviews_status = [] # list of comments (good or bad)
        for review in soup_result:
            if review.string:
                # passing the review to our model
                movie_review_list = np.array([review.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_list.append(review.string)
                reviews_status.append('Good' if pred else 'Bad')
        
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 

        # get movie names for auto completion
        suggestions = get_suggestions()
        
        return render_template('recommend.html',movie=movie,mtitle=r,t='l',reviews=movie_reviews)

if __name__ == '__main__':
    app.run(debug=True)