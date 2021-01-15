import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

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

#test
