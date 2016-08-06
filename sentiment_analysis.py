import tweepy
import json
import numpy
import sys
import os
import re

from prettytable import PrettyTable

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier


MAX_FEATURES = 5000


def pinta_matriz_dispersa(M, nombre_col=None, pre=2):
    filas, columnas = M.shape
    header = nombre_col != None
    pt = PrettyTable(nombre_col, header=header)
    for fila in range(filas):
        vf = M.getrow(fila)
        _, cind = vf.nonzero()
        #f = [vf[0, c] if c in cind else '-' for c in range(columnas)]
        pt.add_row([round(vf[0, c],pre) if c in cind else '-' for c in range(columnas)])
        #print (f)
        #pt.add_row(f)
    return pt


def getLexiconDict(file):
    f = open(file,"r",encoding="utf-8")
    lexicon = {}
    for line in f:
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue
        # print(line.split())
        word,polarity = line.split()
        lexicon[word] = polarity
    f.close()
    return lexicon


def saveTweets(tweet_ids):
    path = "tweets"
    os.makedirs(path, exist_ok=True)

    i = 0
    while True:
        print("Loop", str(i))
        # print("Looking for",tweet_ids[i:i + 100])
        tweets = API.statuses_lookup(tweet_ids[i:i + 100])
        # print("Returned",len(tweets),"tweets")
        for tweet in tweets:
            print("Saving tweet", tweet.id)
            filePath = os.path.join(path, str(tweet.id) + ".json")
            f = open(filePath, "w", encoding="utf-8")
            f.write(json.dumps(tweet._json, indent=1, ensure_ascii=False))
            f.close()

        i += 100
        if i >= len(tweet_ids):
            break


def getData(file):
    data = []
    f = open(file, "r")
    for line in f:
        res = line.strip().split("\t")
        data.append(res)
    f.close()
    return data


def myTokenizer(text):
    tknzr1 = TweetTokenizer(strip_handles=False, reduce_len=True, preserve_case=False)
    return tknzr1.tokenize(text)
    # tr = re.compile('\w+')
    # return tr.findall(text)


def myVectorizer(docs, max):
    # vec = CountVectorizer(tokenizer=myTokenizer, max_features=max)
    norm = None
    use_idf = True
    min_df = 1
    max_df = 1
    sublinear_tf = False
    # vec = TfidfVectorizer(norm=None, smooth_idf=False, stop_words=stop)
    # vec = TfidfVectorizer(norm=norm, use_idf=use_idf, stop_words=stopwords.words("spanish"), max_features=max,
    #                       tokenizer=myTokenizer)
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf,
                          stop_words=stopwords.words("spanish"),
                          max_features=max, tokenizer=myTokenizer, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    voca = vec.get_feature_names()
    return X, voca


def getTweetsData(data):
    valid_tweets = []
    valid_sentiments = []
    tweets_text = []
    for d in data:
        tweet_id = d[0]
        sentiment = d[1] if len(d)>1 else ""
        filePath = os.path.join("tweets", str(tweet_id) + ".json")
        if(os.path.exists(filePath)):
            valid_tweets.append(tweet_id)
            valid_sentiments.append(sentiment)
            f = open(filePath, "r", encoding="utf-8")
            tweet = json.load(f)
            tweets_text.append(tweet["text"])
            f.close()

    return tweets_text, valid_tweets, valid_sentiments


def getExtraFeatures(X, tweets_text, lexicon):
    XX = numpy.zeros((X.shape[0],2))
    # add extra features
    i = 0
    for text in tweets_text:
        tokens = myTokenizer(text)
        positive = 0
        negative = 0
        for token in tokens:
            sentiment = lexicon.get(token)
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
        XX[i][0] = positive
        XX[i][1] = negative
        i += 1
    return XX


def print_limits(API):
    limits = API.rate_limit_status()
    print (sorted(limits['resources'].keys()))
    print (json.dumps(limits['resources']['application'],indent=2))


consumer_key = "Ogq1nyoX6v6jMEM1TyPj02Ira"
consumer_secret = "aFp45adydNJM2YSK9IOrR1Si0lHNV3L0bzKGY7E45heQd0GKcN"
access_token = "4767650120-A7PlKzDbdnZVakTYe7tgc5CAAxbDucPm0YepQPF"
access_token_secret = "y94yceopo6zEnXau0TlwSpDzieGrEj55SQrhwnrY9sjEP"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

API = tweepy.API(auth)

print (API)

# print_limits(API)
# exit()

### Save tweets to file ###

# # training tweets
# data = getData("TASS2014_training_polarity.txt")
# tweet_ids = [x for [x,y] in data]
# saveTweets(tweet_ids)
#
# # minitest tweets
# test_data = getData("TASS2014_training_polarity.txt")
# tweet_ids = [x for [x,y] in test_data]
# saveTweets(tweet_ids)
#
# test tweets
# tweet_ids = getData("TASS2014_test_ids.txt")
# tweet_ids = [x for [x] in tweet_ids]
# i=0
# missing=[]
# for x in tweet_ids:
#     if not os.path.exists("tweets/"+x+".json"):
#         # print(x)
#         i += 1
#         missing.append(x)
# print(i)
# saveTweets(missing)
# saveTweets(tweet_ids)
# exit()


data = getData("TASS2014_training_polarity.txt")
print("Read", len(data), "lines")
print(data[0])

# tweet_ids = [x for [x,y] in data]
lexicon = getLexiconDict("ElhPolar_esV1.lex.txt")
# print(lexicon)


### Training data ###
tweets_text, valid_tweets, valid_sentiments = getTweetsData(data)

print("There are",len(tweets_text),"valid tweets")
print("There are",len(valid_sentiments),"valid sentiments")


### Testing data ###

# test_data = getData("TASS2014_minitest_polarity.txt")
test_data = getData("TASS2014_test_ids.txt")

test_tweets_text, test_valid_tweets, test_valid_sentiments = getTweetsData(test_data)

print("There are", len(test_tweets_text), "valid test tweets")
print("There are", len(test_valid_sentiments), "valid test sentiments")


### Vectorizer ###

print("Feature extraction...")
norm = None
use_idf = True
min_df = 1
max_df = 1
sublinear_tf = False
vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf,
                      stop_words=stopwords.words("spanish"),
                      max_features=MAX_FEATURES, tokenizer=myTokenizer, ngram_range=(1, 2))
X = vec.fit_transform(tweets_text)
test_X = vec.transform(test_tweets_text)
voca = vec.get_feature_names()

# print(voca)
print("The vocabulary has", len(voca), "features")

print("The feature matrix has shape", X.shape)
XX = getExtraFeatures(X, tweets_text, lexicon)
print("The extra features matrix has shape", XX.shape)
X = numpy.concatenate((X.toarray(), XX), axis=1)
print("The final features matrix has shape", X.shape)


print("The test feature matrix has shape", test_X.shape)
test_XX = getExtraFeatures(test_X, test_tweets_text, lexicon)
print("The test extra features matrix has shape", test_XX.shape)
test_X = numpy.concatenate((test_X.toarray(), test_XX), axis=1)
print("The test final features matrix has shape", test_X.shape)


### Classifer ###

# clf = GaussianNB()
# clf = tree.DecisionTreeClassifier()
# clf = RandomForestClassifier(n_estimators=100)
clf = svm.SVC()
print("Training classifier...")
clf.fit(X, valid_sentiments)

print("Predicting...")
preds = clf.predict(test_X)

# print(preds)

# print("There are ", (test_valid_sentiments != preds).sum(), "wrong predictions out of", len(test_valid_sentiments))
# print("Accuracy:", (100.0 * (test_valid_sentiments == preds).sum()) / len(test_X))


### Save results ###

test_tweet_ids = [x for [x] in test_data]
f = open("TASS2014_preds.txt","w")
i = 0
for tweet_id in test_tweet_ids:
    sentiment = "NONE"
    # print(i,tweet_id,tweet_id in test_valid_tweets)
    if tweet_id in test_valid_tweets:
        sentiment = preds[i]
        i += 1
    f.write(tweet_id+"\t"+sentiment+"\n")
f.close()
