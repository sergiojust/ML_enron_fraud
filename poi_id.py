#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester

features_list = ['poi',
                'salary',
                'bonus', 
                'long_term_incentive', 
                'deferred_income', 
                'deferral_payments',
                'loan_advances', 
                'other',
                'expenses', 
                'director_fees',
                'total_payments',
                'exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value',
                'to_messages',
                'from_messages',
                'from_this_person_to_poi',
                'from_poi_to_this_person']


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Create, order and substitute NaNs in a dataframe called "enron" to easily store and access the dataset
enron = pd.DataFrame.from_dict(data_dict, orient = 'index')
enron = enron[features_list]
enron = enron.replace('NaN', np.nan)
enron.info()
enron.describe()

poi_non_poi = enron.poi.value_counts()
poi_non_poi.index=['non-POI', 'POI']
print "POI / non-POI split"
poi_non_poi

print("Number of NaN values in the dataset: "), enron.isnull().sum().sum()

email_features = ['to_messages', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person']

imp = Imputer(strategy='mean', axis=0)

enron.loc[enron[enron.poi == 1].index,email_features] = imp.fit_transform(enron[email_features][enron.poi == 1])
enron.loc[enron[enron.poi == 0].index,email_features] = imp.fit_transform(enron[email_features][enron.poi == 0])

# replace NaN's with 0
enron.iloc[:,:15] = enron.iloc[:,:15].fillna(0)

# complete Sanjay and Robert data with FindLaw data
enron.loc['BHATNAGAR SANJAY','expenses'] = 137864
enron.loc['BHATNAGAR SANJAY','total_payments'] = 137864
enron.loc['BHATNAGAR SANJAY','exercised_stock_options'] = 1.54563e+07
enron.loc['BHATNAGAR SANJAY','restricted_stock'] = 2.60449e+06
enron.loc['BHATNAGAR SANJAY','restricted_stock_deferred'] = -2.60449e+06
enron.loc['BHATNAGAR SANJAY','other'] = 0
enron.loc['BHATNAGAR SANJAY','director_fees'] = 0
enron.loc['BHATNAGAR SANJAY','total_stock_value'] = 1.54563e+07
enron.loc['BELFER ROBERT','total_payments'] = 3285
enron.loc['BELFER ROBERT','deferral_payments'] = 0
enron.loc['BELFER ROBERT','restricted_stock'] = 44093
enron.loc['BELFER ROBERT','restricted_stock_deferred'] = -44093
enron.loc['BELFER ROBERT','total_stock_value'] = 0
enron.loc['BELFER ROBERT','director_fees'] = 102500
enron.loc['BELFER ROBERT','deferred_income'] = -102500
enron.loc['BELFER ROBERT','exercised_stock_options'] = 0
enron.loc['BELFER ROBERT','expenses'] = 3285


features = ["salary", "bonus"]
salary_vs_bonus = featureFormat(data_dict, features)
### plot features
for sample in salary_vs_bonus:
    salary = sample[0]
    bonus = sample[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
features = ["salary", "bonus"]
salary_vs_bonus = featureFormat(data_dict, features)
### plot features
for sample in salary_vs_bonus:
    salary = sample[0]
    bonus = sample[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

## drop 'Total' from the dataframe
enron = enron.drop(['TOTAL'],0)

enron['proportion_to_poi'] = enron['from_this_person_to_poi']/enron['from_messages']

## clean values divided by 0
enron = enron.replace('inf', 0)

clf = DecisionTreeClassifier(random_state = 42)
clf.fit(enron.iloc[:,1:], enron.iloc[:,:1])

features_score = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_score.append([enron.columns[i+1], clf.feature_importances_[i]])
features_score.sort(key=lambda x: x[1], reverse = True)
for score in features_score:
    print score
features_list = [x[0] for x in features_score]
features_list.insert(0, 'poi')

## Decision Tree  (default parameters) 
clf = DecisionTreeClassifier(random_state = 42)
enron_tree = enron[features_list].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, enron_tree, features_list)
tester.main() 

## Gaussian Naive-Bayes (default parameters)
clf = GaussianNB()

## dataset scaling
scaler = StandardScaler()
enron_scaled = enron[features_list]
enron_scaled = scaler.fit_transform(enron_scaled.iloc[:,1:])

## feature selection
features_list_gnb = ['poi'] + range(3)
enron_gnb = pd.DataFrame(SelectKBest(f_classif, k=5).fit_transform(enron_scaled, enron.poi), index = enron.index)

## PCA
pca = PCA(n_components=3)
enron_gnb_pca = pd.DataFrame(pca.fit_transform(enron_gnb),  index=enron.index)
enron_gnb_pca.insert(0, "poi", enron.poi)
enron_gnb_pca = enron_gnb_pca.to_dict(orient = 'index')  

tester.dump_classifier_and_data(clf, enron_gnb_pca, features_list_gnb)
tester.main()

## Random Forest (default)
clf = RandomForestClassifier(random_state = 42)
clf.fit(enron.iloc[:,1:], np.ravel(enron.iloc[:,:1]))

features_score_rf = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_score_rf.append([enron.columns[i+1], clf.feature_importances_[i]])
features_score_rf.sort(key=lambda x: x[1], reverse = True)
features_list = [x[0] for x in features_score_rf]
features_list.insert(0, 'poi')

features_list_rf = features_list[:11]
enron_rf = enron[features_list_rf].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, enron_rf, features_list_rf)
tester.main()

clf = DecisionTreeClassifier(criterion = 'entropy', 
                             min_samples_split = 21,
                             random_state = 42,
                             min_samples_leaf=6)

clf.fit(enron.iloc[:,1:], enron.poi)

features_score_adjtree = []
for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0:
        features_score_adjtree.append([enron.columns[i+1], clf.feature_importances_[i]])
features_score_adjtree.sort(key=lambda x: x[1], reverse = True)

features_list_adjtree = [x[0] for x in features_score_adjtree]
features_list_adjtree.insert(0, 'poi')

enron_adjtree = enron[features_list_adjtree].to_dict(orient = 'index')
tester.dump_classifier_and_data(clf, enron_adjtree, features_list_adjtree)
tester.main() 