import pandas as pd                                                                                                                                                                                   
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
data = pd.read_csv("twitter_data1.csv")
print(data)
features = ["UserID","name","No Of Abuse Report", "No Of Rejected Friend Requests", 
            "No Of Friend Requests Thar Are Not Accepted", "No Of Friends", 
            "No Of Followers", "No Of Likes To Unknown Account",
            "No Of Comments Per Day"]
X = data[features]
y = data["Fake Or Not Category"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#logistic Regression
'''log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
y_pred_log_reg = log_reg_model.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print("Logistic Regression Accuracy:", log_reg_accuracy)
joblib.dump(log_reg_model,"mymodel_log.h5")
# Load and predict with Logistic Regression
loaded_log_reg_model = joblib.load("mymodel_log_reg.h5")
new_data = X_test.iloc[:5]  # Just an example, use actual new data here
log_reg_predictions = loaded_log_reg_model.predict(new_data)
print("Logistic Regression Predictions on new data:", log_reg_predictions)
'''
#Random forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy*100)#rf_accurracy*100
joblib.dump(rf_model,"mymodel_ran.h5")
'''# Load and predict with Random Forest
loaded_rf_model = joblib.load("mymodel_ran.h5")
new_data = X_test.iloc[:1]
rf_predictions = loaded_rf_model.predict(new_data)
print("Random Forest Predictions on new data:", rf_predictions)'''
#Naive_bytes
'''NB_model=GaussianNB()
NB_model.fit(X_train, y_train)
y_pred_NB = NB_model.predict(X_test)
NB_accuracy = accuracy_score(y_test, y_pred_NB)
print("Naive bayes Accuracy:", NB_accuracy)
joblib.dump(NB_model,"mymodel_nav.h5")
# Load and predict with Naive Bayes
loaded_NB_model = joblib.load("mymodel_nb.h5")
new_data = X_test.iloc[:5]  # Just an example, use actual new data here
NB_predictions = loaded_NB_model.predict(new_data)
print("Naive Bayes Predictions on new data:", NB_predictions)
'''
#SVM(support vector machine)(handles both linear and nonlinear classification task)
'''svm_model=SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("svm Accuracy:", svm_accuracy)
joblib.dump(svm_model,"mymodel_sv.h5")
# Load and predict with SVM
loaded_svm_model = joblib.load("mymodel_svm.h5")
new_data = X_test.iloc[:5]  # Just an example, use actual new data here
svm_predictions = loaded_svm_model.predict(new_data)
print("SVM Predictions on new data:", svm_predictions)
'''

