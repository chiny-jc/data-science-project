import json
import pandas as pd

with open('train.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

df.describe()