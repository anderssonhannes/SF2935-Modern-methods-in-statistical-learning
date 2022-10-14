import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import X_train, X_test, y_train, y_test

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))