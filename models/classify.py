#!/bin/python

def train_classifier(X, y, cval, norm, solmethod):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	import numpy as np
	cls = LogisticRegression(penalty = norm, C = cval, solver = solmethod, max_iter =100000)
	#penalty = ['l1', 'l2']
	#C = np.logspace(0, 4, 100)
	#hyperparameters = dict(C = C, penalty = penalty)
	#clf = GridSearchCV(cls, hyperparameters, cv = 4, verbose = 0, scoring = 'accuracy')
	cls.fit(X, y)
	#print('Best penalty:' , cls.best_estimator_.get_params()['penalty'])
	#print('Best C: ', cls.best_estimator_.get_params()['C'])
	return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp_prob = cls.predict_proba(X)
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	#print("  Accuracy on %s  is: %s" % (name, acc))
	return acc, yp_prob, yp
