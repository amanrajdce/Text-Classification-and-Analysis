import numpy as np
import sys
import os
#print(sys.path)
import matplotlib as mpl
import random
mpl.use('Agg')
import matplotlib.pyplot as plt
import models.classify as classify
import models.sentiment as sentimentinterface
from wordcloud import WordCloud,STOPWORDS
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

cwd = os.getcwd()
datap = cwd + "/data/bbc/"

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

class BBCClassifier(object):
    def __init__(self):
        self.clf = None
        self.count_vect = None
        self.classes = ['sport', 'tech', 'business', 'politics', 'entertainment']
        self.read_data()
        self.setup_classifier()

    def read_data(self):
        """
        Read the data and split bw train and dev
        """
        print("Reading data")
        dirs = os.listdir(datap)
        dirs =[d for d in dirs if "README.TXT" not in d]
        # read all the files
        data = []
        label = []
        for d in dirs:
            files = os.listdir(datap+d)
            for fi in files:
                with open(datap+d+'/'+fi, 'r', encoding="utf8", errors='ignore') as f:
                    lines = f.readlines()
                    f.close()
                lines = [l.strip() for l in lines]
                lines = " ".join(lines)
                data.append(lines)
                label.append(self.classes.index(d))

        assert(len(data)==len(label))

        # shuffle the data and split into train and dev set
        random.Random(123).shuffle(data)
        random.Random(123).shuffle(label)
        indx = 0.7*len(data)
        train_data = data[:int(indx)]
        self.trainy = label[:int(indx)]
        dev_data = data[int(indx):]
        self.devy = label[int(indx):]

        print('TfidfVectorizer:')
        self.count_vect = TfidfVectorizer(
            max_features = 200000, min_df = 1, max_df = 1.0,
            tokenizer = LemmaTokenizer(), sublinear_tf = False,
            ngram_range = (1, 3),
            stop_words = frozenset(['the', 'a', 'an', 'i', 'he', 'she',
            'they', 'to', 'of', 'it', 'from']))

        self.trainX = self.count_vect.fit_transform(train_data)
        self.devX = self.count_vect.transform(dev_data)


    def setup_classifier(self):
        print("\nTraining classifier")
        cval = 8
        self.clf = classify.train_classifier(
            self.trainX, self.trainy,
            cval, 'l2','lbfgs'
        )
        print('Training input shape: ' + str(self.trainX.shape))
        print("\nEvaluating")
        #train_acc, train_prob, train_pred = classify.evaluate(
        #    sentiment.trainX, sentiment.trainy, cls, name = 'training data'
        #)
        dev_acc, self.dev_prob, self.dev_pred = classify.evaluate(
            self.devX, self.devy, self.clf, name = 'validation data'
        )
        #print('Train accuracy: ' + str(train_acc))
        print('Dev accuracy: ' + str(dev_acc))

        # generate confidence plots
        print('Generating confidence plots')
        self.generate_conf_plots()

        # generate positive and negative wordclouds
        print('Generating wordcloud')
        feat_names = self.count_vect.get_feature_names()
        for cls in self.classes:
            self.plot_topk_wordcloud(feat_names, 10, cls)


    def predict_statistics(self, input):
        print('Generating predictions and graphs')
        vectorizer = self.count_vect
        c = make_pipeline(vectorizer, self.clf)
        class_names = self.classes
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            input, c.predict_proba, num_features=8, top_labels=1
        )
        #exp.show_in_notebook(text=True)
        exp.save_to_file(cwd+'/app/static/tcgraphs/tc_output.html')
        #exp.as_pyplot_figure()


    def generate_conf_plots(self):
        CONF_TH=0.9
        self.find_correct_incorrect(
            self.dev_prob, self.dev_pred, self.devy, CONF_TH
        )
        x_conf = [0, 1, 2]
        x_nconf = [4, 5, 6]
        scores_conf = [self.conf_perc, self.corr_perc, self.incorr_perc]
        scores_nconf = [self.nconf_perc, self.ncorr_perc, self.nincorr_perc]

        plt.figure(1,figsize=(8, 9))
        plt.bar(
            x_conf+x_nconf, scores_conf+scores_nconf,
            color = ('b', 'g', 'r', 'b', 'g', 'r'), align='edge'
        )
        plt.xticks(
            x_conf + x_nconf,('Confident', 'Confident & \nCorrect',
            'Confident & \n Incorrect', 'Not-Confident',
            'Not-Confident & \nCorrect', 'Not-Confident & \n Incorrect'),
            rotation=30, size=12
        )
        plt.ylabel('Percentage', size=15)
        plt.title(
            'Analysis of confident predictions with confidence threshold='+\
            str(CONF_TH), size=15
        )
        plt.savefig(cwd+'/app/static/tcgraphs/confidence.png', format='png', dpi=100)
        plt.close()


    def plot_topk_wordcloud(self, feat_names, k=10, clas='sport'):
        coefficients = self.clf.coef_[self.classes.index(clas)]
        top_k = np.argsort(coefficients)[-k:]

        top_k_words = []
        for i in top_k:
            #print(feat_names[i], coefficients[i])
            top_k_words.append(feat_names[i])

        words = " ".join(top_k_words)
        # TODO: if required to scale frequency of words as per weights
        # plot wordcloud
        wordcloud = WordCloud(
                stopwords=STOPWORDS, background_color='white',
                width=500, height=500).generate(words)
        plt.figure(1,figsize=(5, 5))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title('Top '+ str(k) + ' ' + clas + ' words', size=15)
        plt.savefig(cwd+'/app/static/tcgraphs/' + clas + '.png', format='png', dpi=200)
        plt.close()


    def find_correct_incorrect(
        self, pred_prob, pred_lb, gt_lb, conf_score=0.8
    ):
        """
        Use given confidence score as threshold for treating a
        given prediction confident. Then, returns the percentage
        of confident samples pedicted correctly and incorrectly.
        """
        confident_pred = []
        nconfident_pred = []
        for idx in range(len(pred_prob)):
            if max(pred_prob[idx]) >= conf_score:
                confident_pred.append(idx)
            else:
                nconfident_pred.append(idx)

        # classifier is confident about predictions on this much percentage
        self.conf_perc = round(100*len(confident_pred)/len(pred_prob), 2)
        self.nconf_perc = 100. - self.conf_perc
        print("Classifier is confident about {}% of predictions".\
            format(self.conf_perc))

        # classifier is correct/incorrect on confident predictions
        confident_corr = 0
        confident_incorr = 0
        for idx in confident_pred:
            if(gt_lb[idx] == pred_lb[idx]):
                confident_corr += 1
            else:
                confident_incorr += 1

        self.corr_perc = round(100*confident_corr/len(confident_pred), 2)
        self.incorr_perc = round(100*confident_incorr/len(confident_pred), 2)
        print("Classifier is correct {}% times on confident predictions".\
            format(self.corr_perc))
        print("Classifier is incorrect {}% times on confident predictions".\
            format(self.incorr_perc))

        # classifier is correct/incorrect on not-confident predictions
        print("Classifier is not confident about {}% of predictions".\
            format(self.nconf_perc))
        nconfident_corr = 0
        nconfident_incorr = 0
        for idx in nconfident_pred:
            if(gt_lb[idx] == pred_lb[idx]):
                nconfident_corr += 1
            else:
                nconfident_incorr += 1

        self.ncorr_perc = round(100*nconfident_corr/len(nconfident_pred), 2)
        self.nincorr_perc = round(100*nconfident_incorr/len(nconfident_pred), 2)
        print("Classifier is correct {}% times on non-confident predictions".\
            format(self.ncorr_perc))
        print("Classifier is incorrect {}% times on non-confident predictions".\
            format(self.nincorr_perc))
