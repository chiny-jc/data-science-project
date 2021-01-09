import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

with open('train.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

df.describe()

np.random.seed(123)
df = df.sample(frac=1) # shuffle data
df_dev, df_test = train_test_split(df, test_size=0.15)
df_train, df_valid = train_test_split(df_dev, test_size=0.15)