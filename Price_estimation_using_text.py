#NOT USED IN FINAL PROJECT SIDE TASK IN LEARNING LINEAR REGRESSION
#Credit to https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer    
from sklearn.linear_model import SGDClassifier
from sklearn import metrics, model_selection
from sklearn.utils import Bunch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups 
from sklearn.model_selection import GridSearchCV


categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

products = pd.read_csv("Products_formated.csv", lineterminator="\n",index_col= 0)
#data = Bunch(inputs = list(products['product_name']),targets = list(round(products['price']))) #extract input and targets into bunch datatype
product_inputs = (products[['product_name','product_description','location']].to_numpy()) #for multiple string inputs values need to be joined into one string
combined_inputs = []
for i in product_inputs:
   combined_inputs.append('\n'.join(i))
data = Bunch(inputs = combined_inputs,targets = list(round(products['price']))) #extract inputs and targets into bunch datatype
X_train, X_test, y_train, y_test = model_selection.train_test_split(data.inputs, data.targets, test_size=0.3) #split data for training and testing


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=5, tol=None))])

parameters = {
     'vect__ngram_range': [(1, 1), (1, 2)],
     'tfidf__use_idf': (True, False),
     'clf__alpha': (1e-2, 1e-3)}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)


gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
# text_clf.fit(twenty_train.data, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
#predicted = text_clf.predict(docs_test)
predicted = gs_clf.predict(docs_test)
print("Best grid search score for twenty train dataset",gs_clf.best_score_)
print("Mean of correctly predicted values for twenty train dataset",np.mean(predicted == twenty_test.target))
print("MSE of twenty train twenty train dataset",metrics.mean_squared_error(twenty_test.target, predicted))


gs_clf = gs_clf.fit(X_train, y_train)
#text_clf.fit(X_train, y_train)
#predicted = text_clf.predict(X_test)
predicted = gs_clf.predict(X_test)
print("Best grid search score for products dataset",gs_clf.best_score_)
print("Mean of correctly predicted values products dataset",np.mean(predicted == y_test))
print("MSE of products dataset ", metrics.mean_squared_error(y_test, predicted))

y_test_upper_range = [x + 5 for x in y_test]
y_test_lower_range = [x - 5 for x in y_test]

in_range = []
for i in range(0,len(predicted)):
    if (predicted[i] > y_test_lower_range[i] and predicted[i] < y_test_upper_range[i]):
        in_range.append(i)
print("predicted values between test values +- 5",len(in_range)/len(predicted))