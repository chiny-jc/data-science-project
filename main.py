''' ------------------------------ IMPORTING THE LIBRARIES -------------------------------- '''
import json
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import SelectFromModel, RFECV


from rfpimp import *
from sklearn.inspection import permutation_importance
from time import time

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
    print(address)
    ''' delete rows that contain descriptions instead of actual addresses '''
    address_delete = [] 
    for i in range(len(df)):
        address_val = df[address][i]
        if re.search('!' or '*', address_val):
            address_delete.append(i)

    df = df.drop(df.index[address_delete])
    print("Num. of deleted addresses: {}".format(
        len(address_delete)))
    
    
    ''' Data Cleaning '''
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

cat_vars = ['building_id','manager_id','display_address','street_address']
OE = OrdinalEncoder()
for cat_var in cat_vars:
    print ("Ordinal Encoding %s" % (cat_var))
    df[cat_var]=OE.fit_transform(df[[cat_var]])


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

v = CountVectorizer(stop_words='english', max_features=50)
x = v.fit_transform(df['features']\
                                     .apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x])))

df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
df.drop('features', axis=1, inplace=True)
df = df.join(df1.set_index(df.index))


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
df = df.drop(['building_id', 'description', 'created', 'display_address', 'manager_id', 'photos', 'street_address', 'listing_id' ], axis=1) 

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

    


np.random.seed(123)
df = df.sample(frac=1) # shuffle data


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
df_dev, df_rest = train_test_split(scaled_df, test_size=0.3)
df_test, df_val = train_test_split(df_rest, test_size=0.5)
'''

X = scaled_df.drop("interest_level", axis=1)
y = scaled_df["interest_level"]

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.15)



'''-------------------------------- Pruning of Decision Tree -------------------------------'''

'''
tree = DecisionTreeClassifier(criterion = "gini", random_state = 123) 
tree = tree.fit(X_train, y_train) 
y_pred = tree.predict(X_test) 
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
    scores = cross_val_score(tree, X_train, y_train, cv=5)
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
tree_pruned = tree_pruned.fit(X_train, y_train)




y_pred = tree_pruned.predict(X_test) 
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 

#plotting the best tree
plt.figure(figsize = (20,17))
plot_tree(tree_pruned, filled = True, rounded = True,class_names = ['Low Interest', 'Medium Interest', 'High Interest'],feature_names = X.columns)[0]
'''


'''---------Random Forest Feature Selection --------------------------------'''


sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_train, y_train)


#To see which feature are important
sel.get_support()


#to get the number of important features
selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)


#to get the name of feature selected
print(selected_feat)


X_train= X_train[selected_feat]
X_test= X_test[selected_feat]


'''---------------------------------RFECV Feature Selection---------------------'''


estimator = RandomForestClassifier()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X_train, y_train)
sel_features = selector.get_support(indices=True)
print(sel_features)

for feat in sel_features:
    sel_feat = X.columns[sel_features]
print(sel_feat)


X_train= X_train[sel_feat]
X_test= X_test[sel_feat]



'''-------------------------------Hyperparameter Tuning the Random Forest in Python-----------------------------------------'''


 # Number of trees in random forest
n_estimators = np.arange(start=100, stop=2001, step=10)
 #Number of features to consider at every split'''
max_features = ['auto', 'sqrt']
 #Maximum number of levels in tree'''
max_depth = np.arange(start=10, stop=111, step=5)
#max_depth.append(None)
 #Minimum number of samples required to split a node'''
min_samples_split = np.arange(start=2, stop=101)
 #Minimum number of samples required at each leaf node'''
min_samples_leaf = np.arange(start=2, stop=101)
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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 5, verbose=2, 
                               random_state=123, n_jobs = -1)

'''Fit the random search model'''
rf_random.fit(X_train, y_train)

print('test')

best_params = rf_random.best_params_

print(best_params)

#Creating the best model

rf = RandomForestClassifier(n_estimators=best_params['n_estimators'], min_samples_split=best_params['min_samples_split'],
                            min_samples_leaf=best_params['min_samples_leaf'], max_features = best_params['max_features'], 
                            max_depth=best_params['max_depth'], bootstrap= best_params['bootstrap'], oob_score = False, 
                            random_state=123)

rf.fit(X_train, y_train)


y_pred = rf.predict_proba(X_test)

print('log_loss:', log_loss(y_test, y_pred))
print('train_acc:', rf.score(X_train, y_train))
print('test_acc:', rf.score(X_test, y_test))





'''-------------------------------Choosing the best model--------------------------------------------'''


ideal_ccp_alpha = 0.000289


#Defining model parameters from the tuned parameter
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': best_params['n_estimators'],
            'min_samples_split': best_params['min_samples_split'],
            'min_samples_leaf': best_params['min_samples_leaf'],
            'max_features': best_params['max_features'],
            'max_depth': best_params['max_depth'],
            'bootstrap': best_params['bootstrap']
        }
    },
    
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'ccp_alpha': ideal_ccp_alpha
        }
    },
    
}

#Loop over both decision tree and random forest to determine the best model for the given dataset
scores = []
print(model_params.items())

trained_models = []
for model_name, mp in model_params.items():
    clf = mp['model']
    clf.set_params(**mp['params'])
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    scores.append({
        'model': model_name,
        'train_score': clf.score(X_train, y_train),
        'test_score': clf.score(X_test, y_test),
        'log_loss': log_loss(y_test, y_pred)
    })
    trained_models.append(clf)
    
best_model = pd.DataFrame(scores, columns= ['model','train_score','test_score', 'log_loss'])
print(best_model)
print(trained_models)




'''------------------------------------ Feature Importance ----------------------------------------------------'''

rf = trained_models[0]


#MDI
MDI_importances = rf.feature_importances_
indices = np.argsort(MDI_importances)
features = X_train.columns

#MDA
MDA_test = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=123)
sorted_idx1 = MDA_test.importances_mean.argsort()
MDA_train = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=123)
sorted_idx2 = MDA_train.importances_mean.argsort()

#Plotting
plt.rcParams["figure.figsize"]=15,5
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title('MDI Importances')
ax1.barh(range(len(indices)), MDI_importances[indices], color='b', align='center')
ax1.set_yticks( np.arange(len(X_train.columns)))
ax1.set_yticklabels(features[indices])
ax1.set(xlabel='Relative Importance')
ax2.boxplot(MDA_train.importances[sorted_idx2].T,
vert=False, labels=features[sorted_idx2])
tpmp=ax2.set_title("Permutation Importances (train)")
ax3.boxplot(MDA_test.importances[sorted_idx1].T,
vert=False, labels=features[sorted_idx1])
tpmp=ax3.set_title("Permutation Importances (test)")
fig.tight_layout()

#Table of results
# record times for comparison
t0 = time()
# Gini importance
gini_imp = pd.DataFrame({'Feature': X.columns, 'Gini Importance': rf.feature_importances_}).set_index('Feature')

t1 = time()# Permutation importance for train
perm_imp = importances(rf, X_train, y_train)

t2 = time()
res= gini_imp.merge(perm_imp, left_index=True, right_index=True).reset_index().\
        rename(columns={'Importance': 'Permutation Importance'})
res.loc[len(X_train.columns)+1] = ['runtime(s)', t1-t0, t2-t1]

print(res)



















