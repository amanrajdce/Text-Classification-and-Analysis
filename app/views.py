"""
Routing module
"""
from app import app
from flask import render_template, flash, request
from .forms import InputTextForm
import nltk
import os
nltk.download('wordnet')
import sys
sys.path.append('../models')
from models.binary_classifier import BinaryClassifier
from models.bbc_classifier import BBCClassifier
cwd = os.getcwd()

# setup binary classifier
print("Setting up Binary classifier...")
binary_clf = BinaryClassifier()
print("Done!!Setting up Binary classifier")

# setup binary classifier
print("Setting up BBC classifier...")
bbc_clf = BBCClassifier()
print("Done!!Setting up BBC classifier")

# Submit button in web for pressed
@app.route('/', methods=['POST'])
@app.route('/index', methods=['POST'])
def manageRequest():

    # some useful initialisation
    theInputForm = InputTextForm()
    userText = "and not leave this empty!"
    typeText = "You should write something ..."
    language = "EN"

      # POST - retrieve all user submitted data
    if theInputForm.validate_on_submit():
        userText = theInputForm.inputText.data
        typeText = "Your own text"

    # DEBUG flash('read:  %s' % typeText)

    # Which kind of user action ?
    if 'TC'  in request.form.values():
        bbc_clf.predict_statistics(userText)
        return render_template('results.html')

    else:
        binary_clf.predict_statistics(userText)
        return render_template('sentiment.html')


  # render web form page
@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def initial():
    # render the initial main page
    return render_template('index.html',
                           title = 'Text Classifier',
                           form = InputTextForm())

@app.route('/results')
def results():
    return render_template('index.html',
                           title='Text Classifier')

  # render about page
@app.route('/about')
def about():
    return render_template('about.html',
                           title='About')
