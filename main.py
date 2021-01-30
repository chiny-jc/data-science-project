''' ------------------------------ IMPORTING THE LIBRARIES -------------------------------- '''
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string 

from re import search 

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet


import functions

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
df['features'].apply(lambda x: functions.add_unique_elements(x, all_unique_features))
print('Num. of Unique Features: {}'.format(len(all_unique_features)))


df['interest_level'] = df['interest_level'].apply(lambda x: 0 if x=='low' else 1 if x=='medium' else 2)
# Check homogeneity of target values
sns.countplot('interest_level', data=df)
plt.title('Unbalanced Classes')

''' --------------------------------- FEATURE ENGINEERING --------------------------------- '''

'''  ----- CATEGORICAL VARIABLES ----- '''

addresses = ['display_address', 'street_address']

for address in addresses:
    address_column = df[address]
    address_column_transformed = ( address_column
                                           .apply(str.upper)
                                           .apply(lambda x: x.replace('WEST','W'))
                                           .apply(lambda x: x.replace('EAST','E'))
                                           .apply(lambda x: x.replace('STREET','ST'))
                                           .apply(lambda x: x.replace('AVENUE','AVE'))
                                           .apply(lambda x: x.replace('BOULEVARD','BLVD'))
                                           .apply(lambda x: x.replace('.',''))
                                           .apply(lambda x: x.replace(',',''))
                                           .apply(lambda x: x.replace('&',''))
                                           .apply(lambda x: x.replace('(',''))
                                           .apply(lambda x: x.replace(')',''))
                                           .apply(lambda x: x.strip())
                                           #.apply(lambda x: re.sub('(?<=\d)[A-Z]{2}', '', x))
                                           .apply(lambda x: re.sub('[^A-Za-z0-9]+ ', '', x)) #remove all special characters and punctuaction
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

    print("Num. of Unique Addresses after Transformation: {}".format(
        address_column_transformed.nunique()))

    df[address] = address_column_transformed 


''' delete rows that contain descriptions instead of actual addresses '''
address_delete = [] 
for i in range(len(df)):
    address_val = df[address][i]
    if search('!', address_val):
        address_delete.append(i)
        
df = df.drop(df.index[address_delete])
  
display=df["display_address"].value_counts()
manager_id=df["manager_id"].value_counts()
building_id=df["building_id"].value_counts()
street=df["street_address"].value_counts()

df["display_count"]=df["display_address"].apply(lambda x:display[x])
df["manager_count"]=df["manager_id"].apply(lambda x:manager_id[x])  
df["building_count"]=df["building_id"].apply(lambda x:building_id[x])
df["street_count"]=df["street_address"].apply(lambda x:street[x])

price_by_building = df.groupby('building_id')['price'].agg([np.min,np.max,np.mean]).reset_index()
price_by_building.columns = ['building_id','min_price_by_building',
                            'max_price_by_building','mean_price_by_building']
df = pd.merge(df,price_by_building, how='left',on='building_id')
df = df.drop(df.index[address_delete])

'''
LBL = LabelEncoder()
LE_vars=[]
for cat_var in cat_vars:
    print ("Label Encoding %s" % (cat_var))
    df[cat_var]=LBL.fit_transform(df[cat_var])

ord_enc = LabelEncoder()
df["manager_id_label"] = ord_enc.fit_transform(df[["manager_id"]])
df[["manager_id_label", "manager_id"]].head()

We still can't run this code, hence manager_id is still not transformed, should we drop it?
'''


'''  ----- TEXT VARIABLES ----- '''

# Studies have shown that titles with excessive all caps and special characters give renters the impression 
# that the listing is fraudulent – i.e. BEAUTIFUL***APARTMENT***CHELSEA.
df['num_of_#']=df.description.apply(lambda x:x.count('#'))
df['num_of_!']=df.description.apply(lambda x:x.count('!'))
df['num_of_$']=df.description.apply(lambda x:x.count('$'))
df['num_of_*']=df.description.apply(lambda x:x.count('*'))
df['num_of_>']=df.description.apply(lambda x:x.count('>'))

df['has_phone'] = df['description'].apply(lambda x:re.sub('['+string.punctuation+']', '', x).split())\
        .apply(lambda x: [s for s in x if s.isdigit()])\
        .apply(lambda x: len([s for s in x if len(str(s))==10]))\
        .apply(lambda x: 1 if x>0 else 0)
df['has_email'] = df['description'].apply(lambda x: 1 if '@renthop.com' in x else 0)

display_address_column = df['description']
df['description'] = [functions.text_cleaner(x) for x in display_address_column]

df['length_description'] = df['description'].apply(lambda x: len(x))
df['num_words_description'] = df['description'].apply(lambda x: len(x.split(" ")))

df['num_features'] = df['features'].apply(len)

v = CountVectorizer(stop_words='english', max_features=100)
x = v.fit_transform(df['features']\
                                     .apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x])))

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
df.drop('features', axis=1, inplace=True)
res = df.join(df1.set_index(df.index))

'''  ----- DATE VARIABLES ----- '''

df['created'] = pd.to_datetime(df['created'])
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day_of_month'] = df['created'].dt.day
df['created_day_of_week'] = df['created'].dt.dayofweek
df['created_hour'] = df['created'].dt.hour

'''  ----- IMAGE VARIABLES ----- '''

df['num_photos'] = df['photos'].apply(len)
df['photos_per_bedroom'] = df[['num_photos','bedrooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)
df['photos_per_bathroom'] = df[['num_photos','bathrooms']].apply(lambda x: x[0]/x[1] if x[1]!=0 else 0, axis=1)


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
        lambda x: functions.calculate_distance_between_coordinates(central_park_coordinates,(x[0],x[1])), axis=1)

wall_street_coordinates = (40.7059692,-74.0099558)
df['distance_to_wall_street'] = df[['latitude','longitude']].apply(
        lambda x: functions.calculate_distance_between_coordinates(wall_street_coordinates,(x[0],x[1])), axis=1)

times_square_coordinates = (40.7567473,-73.9888876)
df['distance_to_times_square'] = df[['latitude','longitude']].apply(
        lambda x: functions.calculate_distance_between_coordinates(times_square_coordinates,(x[0],x[1])), axis=1)


''' ------------------------------------ Correlation of Features and Target --------------------------------------- '''

""" Object columns dropped"""
df = df.drop(['building_id', 'description', 'created', 'display_address', 'manager_id', 'photos', 'street_address' ], axis=1) 

# Convert target values into ordinal values 

df_corr = df.corr()
df_corr_abs = np.abs(df_corr['interest_level'])

df_corr_abs_sort = df_corr_abs.sort_values(ascending = False)
print(df_corr_abs_sort)

sns.set(rc={'figure.figsize':(15.7,10.27)})
sns.heatmap(df.corr())


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

    
''' ------------------------------------ Balanced Dataset ------------------------------------ '''

'''
df['features'].apply(lambda x: functions.add_unique_elements(x, all_unique_features))
print(len(all_unique_features))
print('Num. of Unique Features: {}'.format(len(all_unique_features)))

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_for_manager_ids = one_hot_encoder.fit_transform(pd.DataFrame(df['manager_id']))

multilabel_binarizer = MultiLabelBinarizer()
one_hot_for_features = multilabel_binarizer.fit_transform(df['features'])
'''
'''
# undersampling: reduce sample size for each class so that we have a balanced dataset
interest_2 = len(df.loc[df["interest_level"]==2]) #class with fewest samples 
print(interest_2)
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
'''

'''------------------------------------- Data Normalization ------------------------------'''


df_copy = df.drop("interest_level", axis=1)
scaler = preprocessing.MinMaxScaler()
names = df_copy.columns
d = scaler.fit_transform(df_copy)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()
scaled_df= scaled_df.join(df[['interest_level']].set_index(scaled_df.index))
scaled_df

''' ------------------------------------ DATA MODELING ------------------------------------ '''

'''
df_train, df_rest = train_test_split(df, test_size=0.3)
df_test, df_val = train_test_split(df_rest, test_size=0.5)

#test
df_dev, df_test = train_test_split(df, test_size=0.15)
df_train, df_valid = train_test_split(df_dev, test_size=0.15)
'''

df_dev, df_rest = train_test_split(scaled_df, test_size=0.3)
df_test, df_val = train_test_split(df_rest, test_size=0.5)

#print(df.columns)
#print(df.dtypes)

#X_scaled = scaled_df
#y = df.interest_level
#X_train_scaled, X_test_scaled, y_train , y_test = train_test_split(X_scaled, y, test_size=0.3)

X_val =  df_val.drop("interest_level", axis=1)
y_val = df_val["interest_level"]

X_test = df_test.drop("interest_level", axis=1)
y_test = df_test["interest_level"]

X_dev = df_dev.drop("interest_level", axis=1)
y_dev = df_dev["interest_level"]


X = scaled_df.drop("interest_level", axis=1)
y = scaled_df["interest_level"]

'''------------------------Hyperparameter Tuning of ElasticNet----------------------------'''

'''
# define model
model = ElasticNet(tol=1)
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = np.arange(0, 1, 0.1)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train_scaled, y_train)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
'''


'''-------------------------------- Pruning of Decision Tree -------------------------------'''

'''
tree = DecisionTreeClassifier(criterion = "gini", random_state = 123) 
tree = tree.fit(X_train_scaled, y_train) 
y_pred = tree.predict(X_test_scaled) 
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 



#ploting the tree

from sklearn.tree import plot_tree
plt.figure(figsize = (15,17))
plot_tree(tree, filled = True, rounded = True, class_names = ['Low Interest', 'Medium Interest', 'High Interest'],feature_names = X_scaled.columns)[0]

#determine values for alpha
path = tree.cost_complexity_pruning_path(X_train_scaled, y_train)

#extract different values for alpha
ccp_alphas = path.ccp_alphas 
ccp_alphas = ccp_alphas[:-1]


#create an array to store the resuls of each fold during cross validation
alpha_loop_values = []

# For each candidate value alpha, we will run 5-fold cross validation
# Then we will store the mean and std of the scores(accuracy) for each call
# to cross_val_score in alpha_loop_values...

for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(tree, X_train_scaled, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
#Now we can draw a graph of the means and std of the scores
#for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values, 
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr = 'std',
                   marker = 'o',
                  linestyle ='--')


#Printing the best alpha

column = alpha_results["mean_accuracy"]
max_index = column.idxmax()
max_index


ideal_ccp_alpha = alpha_results.iloc[max_index]['alpha']
#ideal_ccp_alpha 
#Prunning the classification tree

tree_pruned = DecisionTreeClassifier(random_state = 123,
                                    ccp_alpha=ideal_ccp_alpha)
tree_pruned = tree_pruned.fit(X_train_scaled, y_train)




y_pred = tree_pruned.predict(X_test_scaled) 
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 

#plotting the best tree
plt.figure(figsize = (20,17))
plot_tree(tree_pruned, filled = True, rounded = True,class_names = ['Low Interest', 'Medium Interest', 'High Interest'],feature_names = X_scaled.columns)[0]
'''

'''-------------------------------Hyperparameter Tuning the Random Forest in Python-----------------------------------------'''


 # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
 #Number of features to consider at every split'''
max_features = ['auto', 'sqrt']
 #Maximum number of levels in tree'''
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
 #Minimum number of samples required to split a node'''
min_samples_split = [2, 5, 10, 15, 20]
 #Minimum number of samples required at each leaf node'''
min_samples_leaf = [1, 2, 4, 7, 11]
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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=123, n_jobs = -1)

'''Fit the random search model'''
rf_random.fit(X_val, y_val)

print('test')

best_params = rf_random.best_params_

print(best_params)

#Creating the best model

rf = RandomForestClassifier(n_estimators=377, min_samples_split= 15, min_samples_leaf=2, max_features = 'auto', max_depth=50, bootstrap= True, oob_score = True, random_state=123)

rf.fit(X_dev, y_dev)

score = rf.score(X_dev, y_dev)
print('Accurracy for train:',score)

#OOB is the accuracy in trainnig test using oob samlpes
print('OOB score',rf.oob_score_)

y_pred = rf.predict_proba(X_test)

print(log_loss(y_test, y_pred))
print(rf.score(X_dev, y_dev))
print(rf.score(X_test, y_test))








'''---------Ranfom Forest Feature Selection --------------------------------'''

'''
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_dev, y_dev)



#To see which feature are important
sel.get_support()



#to get the number of important features
selected_feat= X_dev.columns[(sel.get_support())]
len(selected_feat)



#to get the name of feature selected
print(selected_feat)



#Testin the results
sel_feat_df = scaled_df[['latitude', 'listing_id', 'longitude', 'price', 'display_count',
'manager_count', 'building_count', 'min_price_by_building',
'max_price_by_building', 'mean_price_by_building', 'length_description',
'num_words_description', 'created_day_of_month', 'created_hour',
'price_per_room', 'price_per_bedroom', 'price_per_bathroom',
'price_per_word_description', 'price_per_length_description',
'price_per_feature', 'price_per_photo', 'distance_to_central_park',
'distance_to_wall_street', 'distance_to_times_square']]




X_scaled = sel_feat_df
y = scaled_df.interest_level
X_train_scaled, X_test_scaled, y_train , y_test = train_test_split(X_scaled, y, test_size=0.3)




rf = RandomForestClassifier()



rf.fit(X_train_scaled, y_train)



score = rf.score(X_train_scaled, y_train)
print('Accurracy for train:',score)



#OOB is the accuracy in trainnig test using oob samlpes
#print('OOB score',rf.oob_score_)



y_pred = rf.predict_proba(X_test_scaled)



print('Log Loss:',log_loss(y_test, y_pred))
print('Train:',rf.score(X_train_scaled, y_train))
print('Test:', rf.score(X_test_scaled, y_test))
'''

'''-------------------------------Choosing the best model--------------------------------------------'''
#Defining model parameters from the tuned parameter
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [100, 600, 1100]
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'alpha': 0.000289
        }
    },
    
    'enet': {
        'model': ElasticNet(tol=1),
        'params': {
            'alpha': 1e-05,
            'l1_ratio': 0.9
        }
    },
    
}

#Loop over both decision tree and random forest to determine the best model for the given dataset
scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
best_model = pd.DataFrame(scores, columns= ['model','best_score','best_params'])
print(best_model)