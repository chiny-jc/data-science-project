import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

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
=======
df['created_year'] = df['created'].dt.year      # ALL RECORDS ARE YEAR 2016
df['created_month'] = df['created'].dt.month    # ALL RECORDS ARE BETWEEN APRIL AND JUNE
df['created_day'] = df['created'].dt.day

''' ------------------------------------ VISUAL EDA --------------------------------------- '''

num_unique_bathroom_values = df['bathrooms'].nunique()
sns.histplot(data=df, x='bathrooms', bins=num_unique_bathroom_values)
sns.boxplot(df['interest_level'], y=df['bathrooms'], order=['low','medium','high'])

num_unique_bedroom_values = df['bedrooms'].nunique()
sns.histplot(data=df, x='bedrooms', bins=num_unique_bedroom_values)
sns.boxplot(df['interest_level'], y=df['bedrooms'], order=['low','medium','high'])

sns.histplot(data=df, x='price', bins=50)
sns.boxplot(df['interest_level'], y=df['price'], order=['low','medium','high'])

sns.histplot(data=df, x='longitude', bins=50)
sns.boxplot(df['interest_level'], y=df['longitude'], order=['low','medium','high'])

sns.histplot(data=df, x='latitude', bins=50)
sns.boxplot(df['interest_level'], y=df['latitude'], order=['low','medium','high'])

df_with_geolocation = df[['longitude','latitude','interest_level']][df['longitude'] != 0]
sns.scatterplot(x=df_with_geolocation['longitude'], y=df_with_geolocation['latitude'], 
                hue=df_with_geolocation['interest_level'], alpha=0.5)

sns.histplot(data=df, x='num_photos', bins=40)
sns.boxplot(df['interest_level'], y=df['num_photos'], order=['low','medium','high'])

sns.histplot(data=df, x='num_features', bins=40)
sns.boxplot(df['interest_level'], y=df['num_features'], order=['low','medium','high'])

sns.histplot(data=df, x='num_words_description', bins=40)
sns.boxplot(df['interest_level'], y=df['num_words_description'], order=['low','medium','high'])

print(min(df['created_month']), max(df['created_month']))
print(df['created_month'].unique())
sns.histplot(data=df, x='created_month', bins=3)
sns.boxplot(df['interest_level'], y=df['created_month'], order=['low','medium','high'])

print(min(df['created_day']), max(df['created_day']))
print(df['created_day'].unique())
sns.histplot(data=df, x='created_day', bins=31)
sns.boxplot(df['interest_level'], y=df['created_day'], order=['low','medium','high'])

''' --------------------------------------------------------------------------------------- '''

print('Num. of Unique Display Addresses: {}'.format(df['display_address'].nunique()))
print('Num. of Unique Manager IDs: {}'.format(df['manager_id'].nunique()))
print('Num. of Unique Interest Levels: {}'.format(df['interest_level'].nunique()))

all_unique_features = set()

def add_unique_elements(list_to_iterate, set_to_add):
    for element in list_to_iterate:
        set_to_add.add(element)
    
df['features'].apply(lambda x: add_unique_elements(x, all_unique_features))
print(len(all_unique_features))
=======
print('Num. of Unique Features: {}'.format(len(all_unique_features)))

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_for_manager_ids = one_hot_encoder.fit_transform(pd.DataFrame(df['manager_id']))

multilabel_binarizer = MultiLabelBinarizer()
one_hot_for_features = multilabel_binarizer.fit_transform(df['features'])

'''Transforming the display address column'''

display_address_column = df['display_address']
display_address_column_transformed = ( display_address_column
                                       .apply(str.upper)
                                       .apply(lambda x: x.replace('WEST','W'))
                                       .apply(lambda x: x.replace('EAST','E'))
                                       .apply(lambda x: x.replace('STREET','ST'))
                                       .apply(lambda x: x.replace('AVENUE','AVE'))
                                       .apply(lambda x: x.replace('.',''))
                                       .apply(lambda x: x.replace(',',''))
                                       .apply(lambda x: x.strip())
                                       .apply(lambda x: re.sub('(?<=\d)[A-Z]{2}', '', x))
                                       .apply(lambda x: x.replace('FIRST','1'))
                                       .apply(lambda x: x.replace('SECOND','2'))
                                       .apply(lambda x: x.replace('THIRD','3'))
                                       .apply(lambda x: x.replace('FOURTH','4'))
                                       .apply(lambda x: x.replace('FIFTH','5'))
                                       .apply(lambda x: x.replace('SIXTH','6'))
                                       .apply(lambda x: x.replace('SEVENTH','7'))
                                       .apply(lambda x: x.replace('EIGHTH','8'))
                                       .apply(lambda x: x.replace('EIGTH','8'))
                                       .apply(lambda x: x.replace('NINTH','9'))
                                       .apply(lambda x: x.replace('TENTH','10'))
                                       .apply(lambda x: x.replace('ELEVENTH','11'))
                                     )

print(display_address_column_transformed.nunique())

def regex_lookup(column, regex_pattern, match_only=True):
    matches = []
    idx = []
    for i in range(len(column) - 1):
        match = re.search(regex_pattern, column.iloc[i])
        if match != None:
            matches.append(match.group(0))
            idx.append(i)
    
    if match_only:
        return matches
    else:
        return column.iloc[idx]
    
# Convert target values into ordinal values 
df['interest_level'] = df['interest_level'].apply(lambda x: 0 if x=='low' else 1 if x=='medium' else 2)

# Check homogeneity of target values
sns.countplot('interest_level', data=df)
plt.title('Unbalanced Classes')

# undersampling: reduce sample size for each class so that we have a balanced dataset
interest_2 = len(df.loc[df["interest_level"]==2]) #class with fewest samples 

shuffled_df = df.sample(frac=1,random_state=42)
df_0_reduced = shuffled_df.loc[shuffled_df['interest_level'] == 0].sample(n=interest_2,random_state=42)
df_1_reduced = shuffled_df.loc[shuffled_df['interest_level'] == 1].sample(n=interest_2,random_state=42)
df_2 = shuffled_df.loc[shuffled_df['interest_level'] == 2]

# Concatenate dataframes again
df = pd.concat([df_0_reduced, df_1_reduced, df_2]) #balanced dataset

#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('interest_level', data=df)
plt.title('Balanced Classes')
plt.show()

'''Splitting the dataset for modeling'''

np.random.seed(123)
df = df.sample(frac=1) # shuffle data
df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_val = train_test_split(df_rest, test_size=0.5)

#test
=======
df_dev, df_test = train_test_split(df, test_size=0.15)
df_train, df_valid = train_test_split(df_dev, test_size=0.15)

