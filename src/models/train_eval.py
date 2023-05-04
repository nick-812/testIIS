import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pickle import dump
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/current_data.csv", sep=",", header=0)

cevovod = Pipeline ([
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ("preprocess", SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler',StandardScaler()),
    ("MLPR", MLPRegressor())
])

parametri = {
    "MLPR__hidden_layer_sizes": [(64),(32),(16)],
    "MLPR__learning_rate_init": [0.001, 0.01]
}

y = df['final_grade']
df = df.drop(['final_grade'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.3, random_state=1234)

gsModel = GridSearchCV (cevovod, parametri, verbose=2)
gsModel.fit(X_train, y_train)


y_out = gsModel.predict(X_test)

mae=mean_absolute_error(y_test, y_out)
mse=mean_squared_error(y_test, y_out)
evs=explained_variance_score(y_test, y_out)

print(mae)

dump(gsModel, open('models/model.pkl', 'wb'))
