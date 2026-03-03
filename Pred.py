import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

file_path = 'Concrete_data_simplified.csv'
data = pd.read_csv(file_path)

X = data[['Cement','BlastFurnaceSlag','FlyAsh','Water','Superplasticizer','CoarseAggregate','FineAggregate','Age']]
y = data['Concretecompressivestrength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rkf = RepeatedKFold(n_splits=5,n_repeats=5,random_state=42)

# using linear regression
lr_model = LinearRegression()
lr_model.fit(X=X_train,y=y_train)

lr_scores = cross_val_score(estimator=lr_model,X=X_train,y=y_train,cv=rkf,scoring='neg_mean_absolute_error')

print("----LINEAR REGRESSION----")
print(f"Negative MAE scores: {lr_scores}")
print(f"Mean cross-validation: {np.mean(lr_scores):.4f}")
print(f"Standard deviation: {np.std(lr_scores):.4f}")

# using random forest
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X=X_train,y=y_train)

rf_scores = cross_val_score(estimator=rf_regressor,X=X_train,y=y_train,cv=rkf,scoring='neg_mean_absolute_error',n_jobs=-1)

print("----RANDOM FOREST----")
print(f"Negative MAE scores: {rf_scores}")
print(f"Mean cross-validation: {np.mean(rf_scores):.4f}")
print(f"Standard deviation: {np.std(rf_scores):.4f}\n")

# prediction results
y_pred_rf = rf_regressor.predict(X_test)
test_mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"Test MAE (RF): {test_mae_rf:.4f}")

y_pred_lr = lr_model.predict(X_test)
test_mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Test MAE (LR): {test_mae_lr:.4f}")

# with open('rf_regressor.pkl', 'wb') as model_file:
#      pickle.dump(rf_regressor, model_file)

#joblib.dump(rf_regressor, 'rf_model.joblib')