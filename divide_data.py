import pandas as pd
import numpy as np


df = pd.read_csv ('movies_dataset.csv')

train, validate, test = np.split(df.sample(frac=1, random_state=42),
                       [int(.6*len(df)), int(.8*len(df))])

train.to_csv(r'C:\Users\nerhk\PycharmProjects\hackathon\train_capuchon.csv', index = False)
validate.to_csv(r'C:\Users\nerhk\PycharmProjects\hackathon\validate_capuchon.csv', index = False)
test.to_csv(r'C:\Users\nerhk\PycharmProjects\hackathon\test_capuchon.csv', index = False)

# create summary:
summary = train.describe(include="all")
summary.to_csv(r'C:\Users\nerhk\PycharmProjects\hackathon\summary_capuchon.csv', index = True)
