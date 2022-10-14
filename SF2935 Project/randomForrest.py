from preprocessing import preprocess
import pandas as pd

from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

X_df, y_df = preprocess(onehot=False)

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=None)

scaler = StandardScaler()
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())

# param_grid = {'criterion':['gini','entropy'],'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],}
param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [4,5,6,7,8,9,10,11,12,15,20,30],
    'max_features': [4,5,6,7,8,9,10,11,12,13,14,15],
}

# search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)

# X_trans = scaler.fit_transform(X_train)
# search.fit(X_trans, y_train)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

pipe = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200,max_depth=10,criterion='entropy',max_features=9))
scores = cross_val_score(pipe, X_train,y_train,cv=10)
print(mean(scores))

pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))

X_TEST = pd.read_csv("project_test.csv")
y_pred = pipe.predict(X_TEST)
print(y_pred)