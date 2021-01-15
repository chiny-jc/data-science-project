
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV

'''Reading the file'''

with open('data/train.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

'''Exploratory Data Analysis'''

df_head = df.head()
df_describe = df.describe()
df_info = df.info()

df['num_photos'] = df['photos'].apply(len)
df['num_features'] = df['features'].apply(len)
df['num_words_description'] = df['description'].apply(lambda x: len(x.split(" ")))

df['created'] = pd.to_datetime(df['created'])
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day


ord_enc = LabelEncoder()
df["manager_id_label"] = ord_enc.fit_transform(df[["manager_id"]])
df[["manager_id_label", "manager_id"]].head()




print(df['display_address'].nunique())
print(df['manager_id'].nunique())
print(df['interest_level'].nunique())

all_unique_features = set()

def add_unique_elements(list_to_iterate, set_to_add):
    for element in list_to_iterate:
        set_to_add.add(element)
    
df['features'].apply(lambda x: add_unique_elements(x, all_unique_features))
print(len(all_unique_features))

'''Splitting the dataset for modeling'''

np.random.seed(123)
df = df.sample(frac=1) # shuffle data

df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_val = train_test_split(df_rest, test_size=0.5)


'''Hyperparameter Tuning the Random Forest in Python'''
print(df.columns)


num_feats = ["manager_id_label","bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_words_description",
             "created_year", "created_month", "created_day"]


X = df[num_feats]
y = df["interest_level"]
X.head()


 # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
 #Number of features to consider at every split'''
max_features = ['auto', 'sqrt']
 #Maximum number of levels in tree'''
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
 #Minimum number of samples required to split a node'''
min_samples_split = [2, 5, 10]
 #Minimum number of samples required at each leaf node'''
min_samples_leaf = [1, 2, 4]
#Method of selecting samples for training each tree'''
bootstrap = [True, False]


'''Create the random grid'''
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)



''''First create the base model to tune'''

rf = RandomForestClassifier()

''' Use the random grid to search for best hyperparameters'''
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

'''Fit the random search model'''
rf_random.fit(X, y)

print('test')

best_params = rf_random.best_params_

print(best_params)

#Creating the best model

rf = RandomForestClassifier(n_estimators=1000, min_samples_split= 2, min_samples_leaf=2, max_features = 'sqrt', max_depth=60, bootstrap= True, oob_score = True,random_state=0)

rf.fit(X, y)

score = rf.score(X, y)
print('Accurracy for train:',score)

#OOB is the accuracy in trainnig test using oob samlpes
print('OOB score',rf.oob_score_)







