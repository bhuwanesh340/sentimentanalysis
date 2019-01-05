
from flask import Flask, render_template, request
import sys
#sys.path.append('C:\\Users\\Bhuwanesh\\Anaconda3\\Lib\\site-packages')

app = Flask(__name__)

@app.route('/')
def product():
    return render_template('sentiment_home_page.html')

@app.route('/', methods = ['POST'])
def my_form_post():
    review = request.form['review']
    sentiment_op = main_function(review)
    return predict_review_sentiment(sentiment_op)

@app.route('/sentiment_op/')
def predict_review_sentiment(user_review):
    #global words
    #sentiment = nb.predict(vect.transform([user_review]))
    if user_review[0] == 1:
        #img = Image.open('G:\\NU MATERIAL\\NU-TERM2\\BIG_DATA\\PROJECT\\happy.jpeg')
        #img.show()
        return render_template('sentiment_op_happy.html', review_op = user_review[0])
    
    else:
        #img = Image.open('G:\\NU MATERIAL\\NU-TERM2\\BIG_DATA\\PROJECT\\sad.jpeg')
        #img.show()
        return render_template('sentiment_op_sad.html', review_op = user_review[0])



def main_function(review_word):
	import math
	import random
	from collections import defaultdict
	from pprint import pprint
	from collections import Counter
	from nltk.corpus import stopwords
	import re
	import string
	import nltk

	# Prevent future/deprecation warnings from showing in output
	import warnings
	warnings.filterwarnings(action='ignore')

	import seaborn as sns
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd

	from PIL import Image                                                                                


	# In[2]:

	#data=pd.read_csv('G:\\NU MATERIAL\\NU-TERM2\\BIG_DATA\\PROJECT\\hotel-reviews\\7282_1.csv')
	data=pd.read_csv('7282_2.csv')
	#data=pd.read_csv('/media/bhuwanesh-ug1-3317/NIIT/NU MATERIAL/NU-TERM2/BIG_DATA/PROJECT/hotel-reviews/7282_1.csv')


	# In[3]:

	review=pd.DataFrame(data.groupby('reviews.rating').size().sort_values(ascending=False).rename('No of Users').reset_index())

	df=data[['reviews.text','reviews.rating']]

	# In[6]:

	df=df.dropna()
	df[df['reviews.rating'] != 3]
	df['labels'] = np.where(df['reviews.rating'] > 2, 1, 0)


	stop = set(stopwords.words('english'))


	# In[12]:

	def clean_document(doco):
		punctuation = string.punctuation
		punc_replace = ''.join([' ' for s in punctuation])
		doco_link_clean = re.sub(r'http\S+', '', doco)
		doco_clean_and = re.sub(r'&\S+', '', doco_link_clean)
		doco_clean_at = re.sub(r'@\S+', '', doco_clean_and)
		doco_clean = doco_clean_at.replace('-', ' ')
		doco_alphas = re.sub(r'\W +', ' ', doco_clean)
		trans_table = str.maketrans(punctuation, punc_replace)
		doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')])
		doco_clean = doco_clean.split(' ')
		p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
		doco_clean = ([p.sub("", x).strip() for x in doco_clean])
		doco_clean = [word.lower() for word in doco_clean if len(word) > 2]
		doco_clean = ([i for i in doco_clean if i not in stop])
	#     doco_clean = [spell(word) for word in doco_clean]
	#     p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
		doco_clean = ([p.sub("", x).strip() for x in doco_clean])
	#     doco_clean = ([spell(k) for k in doco_clean])
		return doco_clean


	# In[13]:

	review_clean = [clean_document(doc) for doc in df['reviews.text']]
	sentences = [' '.join(r) for r in review_clean]


	# In[14]:

	df['cleantext']=sentences


	# In[15]:

	def top_words(data):
			words_list = data.split(' ')
			counts = Counter(words_list)
			top_words = counts.most_common(20)
			length_of_list = len(top_words)
			index = np.arange(length_of_list)
			


	# In[16]:

	train_positive_sentiment = df.loc[df["labels"] == 1]
	positive_words = ' '.join(train_positive_sentiment['cleantext'])
	#print("Top words in Positive Sentiment")
	top_words(positive_words)


	# In[17]:

	train_positive_sentiment = df.loc[df["labels"] == 0]
	positive_words = ' '.join(train_positive_sentiment['cleantext'])
	#print("Top words in Negative Sentiment")
	top_words(positive_words)


	# In[18]:

	from sklearn.model_selection import train_test_split

	X = df.cleantext
	y = df["labels"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


	# In[19]:

	from sklearn.feature_extraction.text import CountVectorizer

	vect = CountVectorizer(max_features=1000, binary=True)

	X_train_vect = vect.fit_transform(X_train)

	# In[21]:

	from imblearn.over_sampling import SMOTE

	sm = SMOTE()
	X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)


	# In[22]:

	unique, counts = np.unique(y_train_res, return_counts=True)


	from sklearn.naive_bayes import MultinomialNB
	nb = MultinomialNB()
	nb.fit(X_train_res, y_train_res)
	nb.score(X_train_res, y_train_res)


	# In[24]:

	X_test_vect = vect.transform(X_test)
	y_pred = nb.predict(X_test_vect)

	# In[25]:

	from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

	# In[27]:

	from sklearn.linear_model import LogisticRegression
	model1 = LogisticRegression()
	model1.fit(X_train_res, y_train_res)
	ypred=model1.score(X_train_res, y_train_res)
	
	sentiment = nb.predict(vect.transform([review_word]))
	return(sentiment)
	
	

if __name__ == '__main__':
    app.run(debug=True)
	
