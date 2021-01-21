''' ------------------------------ IMPORTING THE LIBRARIES -------------------------------- '''

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

''' -------------------------------------- FUNCTIONS -------------------------------------- '''

def add_unique_elements(list_to_iterate, set_to_add):
    for element in list_to_iterate:
        set_to_add.add(element)
        
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
    
def calculate_distance_between_coordinates(first_coordinates, second_coordinates):
    meters_per_coordinate_degree = 111.139
    latitude_difference_in_km = abs(first_coordinates[0] - second_coordinates[0]) * meters_per_coordinate_degree
    longitude_difference_in_km = abs(first_coordinates[1] - second_coordinates[1]) * meters_per_coordinate_degree
    linear_distance = np.sqrt(latitude_difference_in_km ** 2 + longitude_difference_in_km ** 2)
    return linear_distance

''' ---------------------------------- READING THE FILE ----------------------------------- '''

with open('data/train.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

''' ------------------------------- EXPLORATORY DATA ANALYSIS ----------------------------- '''

df_head = df.head()
df_describe = df.describe()
df_info = df.info()

print('Num. of Unique Display Addresses: {}'.format(df['display_address'].nunique()))
print('Num. of Unique Manager IDs: {}'.format(df['manager_id'].nunique()))

all_unique_features = set()    
df['features'].apply(lambda x: add_unique_elements(x, all_unique_features))
print('Num. of Unique Features: {}'.format(len(all_unique_features)))

''' --------------------------------- FEATURE ENGINEERING --------------------------------- '''

'''  ----- CATEGORICAL VARIABLES ----- '''

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_for_manager_ids = one_hot_encoder.fit_transform(pd.DataFrame(df['manager_id']))

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

print("Num. of Unique Display Addresses after Transformation: {}".format(
    display_address_column_transformed.nunique()))

'''  ----- TEXT VARIABLES ----- '''

df['length_description'] = df['description'].apply(lambda x: len(x))
df['num_words_description'] = df['description'].apply(lambda x: len(x.split(" ")))
df['num_features'] = df['features'].apply(len)

multilabel_binarizer = MultiLabelBinarizer()
one_hot_for_features = multilabel_binarizer.fit_transform(df['features'])

'''  ----- DATE VARIABLES ----- '''

df['created'] = pd.to_datetime(df['created'])
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day_of_month'] = df['created'].dt.day
df['created_day_of_week'] = df['created'].dt.dayofweek
df['created_hour'] = df['created'].dt.hour

'''  ----- IMAGE VARIABLES ----- '''

df['num_photos'] = df['photos'].apply(len)

'''  ----- NUMERICAL VARIABLES ----- '''

df['total_rooms'] = df['bathrooms'] + df['bedrooms']
df['price_per_room'] = df[['price','total_rooms']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)
df['price_per_bedroom'] = df[['price','bedrooms']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)
df['price_per_bathroom'] = df[['price','bathrooms']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)

df['price_per_word_description'] = df[['price','num_words_description']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)
df['price_per_length_description'] = df[['price','length_description']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)
df['price_per_feature'] = df[['price','num_features']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)
df['price_per_photo'] = df[['price','num_photos']].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0, axis=1)

central_park_coordinates = (40.7799963,-73.970621)
df['distance_to_central_park'] = df[['latitude','longitude']].apply(
        lambda x: calculate_distance_between_coordinates(central_park_coordinates,(x[0],x[1])), axis=1)

wall_street_coordinates = (40.7059692,-74.0099558)
df['distance_to_wall_street'] = df[['latitude','longitude']].apply(
        lambda x: calculate_distance_between_coordinates(wall_street_coordinates,(x[0],x[1])), axis=1)

times_square_coordinates = (40.7567473,-73.9888876)
df['distance_to_times_square'] = df[['latitude','longitude']].apply(
        lambda x: calculate_distance_between_coordinates(times_square_coordinates,(x[0],x[1])), axis=1)

''' ------------------------------------ VISUAL EDA --------------------------------------- '''
'''
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
'''
    
''' ------------------------------------ DATA MODELING ------------------------------------ '''

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

np.random.seed(123)
df = df.sample(frac=1) # shuffle data
df_dev, df_test = train_test_split(df, test_size=0.15)
df_train, df_valid = train_test_split(df_dev, test_size=0.15)