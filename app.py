from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from rnn import test_model

# load the model from disk

app = Flask(__name__,template_folder='template')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		message = request.form['message']
		question = [f" {message} "]
		my_prediction = test_model(question)
	return render_template("result.html",prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)