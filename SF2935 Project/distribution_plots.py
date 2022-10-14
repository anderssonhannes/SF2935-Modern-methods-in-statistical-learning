import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

data = pd.read_csv("project_train.csv")



y = data["Label"]
y.to_frame()
X = data.drop("Label",axis=1)


# Change label with one hot encoder
X_key = X['key']
X_key.to_frame()
labelencoder = LabelEncoder()
X_key_cat = labelencoder.fit_transform(X_key)
X_key_cat = pd.DataFrame(X_key_cat,columns=['key_cat'])
X_key = pd.concat([X_key,X_key_cat],axis=1)

enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(X_key['key_cat'].to_frame()).toarray(),columns=['C','Cm','D','Dm','E','F','Fm','G','Gm','A','Am','B'])
X = X.drop('key',axis=1)
X = X.join(enc_df)


# Removing bad data
id1 = X[(X["energy"] > 1)].index
X = X.drop(id1,axis=0)
y = y.drop(id1)
id2 = X[(X["loudness"] < -60)].index
X = X.drop(id2,axis=0)
y = y.drop(id2)
id3 = X[(X["loudness"] > 0)].index
X_df = X.drop(id3,axis=0)
y_df = y.drop(id3)

X_df = X_df.reset_index(drop=True)
y_df = y_df.reset_index(drop=True)


y_1 = y_df[(y_df[:] == 1)]
y_2 = y_df[(y_df[:] == 0)]

X_1 = X_df.iloc[y_1.index]
X_2 = X_df.iloc[y_2.index]


X_1.hist(bins=30)
plt.show()

X_2.hist(bins=30)
plt.show()


