import pandas as pd
import joblib
df = pd.read_csv('app/data.csv')


"""#Model Training"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

y = df['default_status']
x = df.drop(columns=['default_status'])



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

"""Dealing with Imbalance using SMOTE"""

x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_train, y_train)

"""Scaling"""

stsc=StandardScaler()

xtrain=stsc.fit_transform(x_resampled)
xtest=stsc.transform(x_test)

"""#RandomForest"""

rf=RandomForestClassifier(n_estimators=200,random_state=42)

rf.fit(xtrain,y_resampled)

rf.score(xtest,y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on test data
y_pred_rf = rf.predict(xtest)

# Evaluate
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

"""#XgBoost"""

xg=XGBClassifier(
    n_estimators=200,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42)

xg.fit(xtrain,y_resampled)

y_pred = xg.predict(xtest)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""#Exporting Model for Deployment"""

joblib.dump(rf, 'models/random_forest_model.pkl')
joblib.dump(xg, 'models/xgboost_model.pkl')
joblib.dump(stsc, 'models/scaler.pkl')