from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
from flask import Flask, request, render_template, Response
import os
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
pd.set_option('display.max_colwidth', 2500)
pd.set_option('display.max_rows', 5)

app = Flask(__name__)

# Import Data
# train = pd.read_csv("E:\UG\1111FIX SKRIPSI PROGRAM\Skripsi NLP\ds_mamikos1.csv", encoding="ISO-8859-1")
train = pd.read_csv(
    "E:/UG/1111FIX SKRIPSI PROGRAM/Skripsi NLP/ds_mamikos1.csv")
test_pd = pd.DataFrame(train)
x_train = train["Ulasan"]
y_train = train["Label"]

# Split Data
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    x_train, y_train, test_size=0.3, random_state=37)

# Create DataFrame Train
df_train = pd.DataFrame()
df_train['trainx'] = Train_X
df_train['trainy'] = Train_Y
# Create DataFrame Test
df_test = pd.DataFrame()
df_test['testx'] = Test_X
df_test['testy'] = Test_Y
# Create variable Train & Test
trainx = df_train['trainx']
testx = df_test['testx']

# CASE FOLDING & REMOVE REGEX


def preprocess(text):
    clean_data = []
    for x in (text[:]):  # this is Df_pd for Df_np (text[:])
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text)  # remove punc.
        new_text = re.sub(r'\d+', '', new_text)  # remove numbers
        new_text = new_text.lower()  # lower case, .upper() for upper
        if new_text != '':
            clean_data.append(new_text)
    return clean_data


# Case Folding variable Train & Test
CaseFolding_trainx = preprocess(trainx)
CaseFolding_tesx = preprocess(testx)
# Create Column Case Folding Train & Test
df_train['case_folding'] = CaseFolding_trainx
df_test['case_folding'] = CaseFolding_tesx

# TOKENIZATION


def identify_tokens(row):
    review = row['case_folding']
    tokens = word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words


# Data Train Tokenization
df_train['token'] = df_train.apply(identify_tokens, axis=1)
tokens_trainx = df_train['token']
# Data Test Tokenization
df_test['token'] = df_test.apply(identify_tokens, axis=1)
tokens_testx = df_test['token']

# REMOVE STOPWORDS
stops = set(stopwords.words("indonesian"))


def remove_stops(row):
    my_list = row['token']
    meaningful_words = [w for w in my_list if not w in stops]
    return (meaningful_words)


# Data Train Stop Words Removal
df_train['stopwords'] = df_train.apply(remove_stops, axis=1)
stopword_trainx = df_train['stopwords']
# Data Test Stop Words Removal
df_test['stopwords'] = df_test.apply(remove_stops, axis=1)
stopword_testx = df_test['stopwords']

# STEMMING


def stem_list(row):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    my_list = row['stopwords']
    stemmed_list = [stemmer.stem(word) for word in my_list]
    return (stemmed_list)


# Data Train Stemming
df_train['stemming'] = df_train.apply(stem_list, axis=1)
stem_trainx = df_train['stemming']
df_train['final'] = stem_trainx.astype(str)  # array list ke string
text_final_trainx = df_train['final']  # final column

# Data Test Stemming
df_test['stemming'] = df_test.apply(stem_list, axis=1)
stem_testx = df_test['stemming']
df_test['final'] = stem_testx.astype(str)  # array list ke string
text_final_testx = df_test['final']  # final column

# Data Train & Test After Preprocessing
text_final_trainy = df_train['trainy']
text_final_testy = df_test['testy']

text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])

# train the model
text_clf.fit(text_final_trainx, text_final_trainy)
# Predict the test cases
predicted = text_clf.predict(text_final_testx)
# Create Column Prediction Result
df_test['prediksi'] = predicted
# Predictioan Accuracy
akurasi = (accuracy_score(text_final_testy, predicted)*100)
kesalahan = 100-(akurasi)
print(akurasi)


@app.route("/")
@app.route("/index")
def index():
    trainNew = test_pd
    label = test_pd['Label']
    pos = label[label == 'positif']
    neg = label[label == 'negatif']
    return render_template('home.html', tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/casefolding")
def casedata():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(CaseFolding_tesx, name='Case Folding').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('casefolding.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/token")
def tokendata():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(tokens_testx, name='Tokenization').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('token.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/stopwords")
def stopwordsdata():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(
        stopword_testx, name='Stop Words Removal').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('stopwords.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/stemming")
def stemmingdata():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(stem_testx, name='Stemming').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('stemming.html',  tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/sentiment")
def sentimentData():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(predicted, name='Hasil Sentimen').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    return render_template('result.html', tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')])


@app.route("/statistiksentimen")
def statistiksen():
    trainNewA = pd.Series(testx, name='Data Test').reset_index()
    trainNewB = pd.Series(predicted, name='Prediction').reset_index()
    trainNew = pd.concat([trainNewA, trainNewB], axis=1)
    trainNew = trainNew.drop("index", axis=1)
    label = df_test['prediksi']
    pos = label[label == 'positif']
    neg = label[label == 'negatif']
    jml_pos = pos.count()
    jml_neg = neg.count()
    jml_label = label.count()
    persen_pos = str(round((jml_pos/jml_label)*100, 2))
    persen_neg = str(round((jml_neg/jml_label)*100, 2))
    labels = ["Positif", "Negative"]
    data = [jml_pos, jml_neg]
    colors = ["#00a65a", "#dd4b39"]
    img = os.path.join('static', 'img')
    imgNeg = os.path.join(img, 'neg.png')
    imgPos = os.path.join(img, 'pos.png')
    return render_template('statistiksentimen.html', jml_pos=jml_pos, jml_neg=jml_neg, akurasi=akurasi, kesalahan=kesalahan, imgNeg=imgNeg, imgPos=imgPos, tables=[trainNew.to_html(classes='table table-hover table-bordered', header='true', justify='justify', table_id='tabel')], set=zip(data, labels, colors))


@app.route('/sentimenttest')
def sentimentTest():
    return render_template('sentiment.html')


@app.route('/checksentiment', methods=['POST'])
def checkSentiment():
    text = request.form['text']
    hasil = text_clf.predict([text])
    if hasil == ['negatif']:
        hasil = 'Ulasan yang anda input sentimen (-) Negatif'
    else:
        hasil = 'Ulasan yang anda input sentimen (+) Positif'
    return render_template('sentimenthasil.html', hasil=hasil, text=text)


if __name__ == '__main__':
    app.run(debug=True)
