from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re, string
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer



model = pickle.load(open('minor_project_model.pkl', 'rb'))
tfidf = pickle.load(open('minor_project_vectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    # return "Hello world"
    return render_template('index.html')

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt')


def preprocess(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '',  tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'ð', '', tweet)
    # tweet_tokens = word_tokenize(tweet)
    # filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    
    return " ".join(tweet)


# def lemmatizing(data):
#   tweet = [lemmatizer.lemmatize(word) for word in data]
#   return data


@app.route('/predict', methods=['POST'])
def prediction():
    lyrics = request.form.get('lyrics')

    # preprocess
    plyrics = preprocess(lyrics)
    # lem_txt = lemmatizing(plyrics)
    vectorized = tfidf.transform([plyrics]).toarray()[0]

    # # prediction
    p = model.predict(np.expand_dims(vectorized, axis=0))
    # result = 0
    if p[0] == 0:
        return lyrics+":   Not Hate"
    elif p[1]==1:
        return lyrics+":   Hate"
    return str(p)

    # return str(vectorized)


if __name__ == '__main__':
    app.run(debug=True)