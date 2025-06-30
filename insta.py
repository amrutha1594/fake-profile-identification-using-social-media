import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("test.csv")

# Check the data
print(data.head())  # Print first few rows to inspect data
data.info()  # Get information about columns and null values

# Data preprocessing
# Assuming there are no null values for simplicity
# If there are null values, handle them appropriately (e.g., data.fillna(method='ffill'))

# Define features (x) and target (y)
x = data[["profile pic", "fullname words", "nums/length fullname", "name==username",
          "description length", "external URL", "private", "#posts",
          "#followers", "#follows"]]  # Features
y = data["fake"]  # Target


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression model
'''
logreg_model = LogisticRegression()
logreg_model.fit(x_train, y_train)
y_pred_logreg = logreg_model.predict(x_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", accuracy_logreg)
classification_report_logreg = classification_report(y_test, y_pred_logreg)
print("Logistic Regression Classification Report:\n", classification_report_logreg)
joblib.dump(logreg_model,"mymodel_log.h5")
'''
# Random Forest Classifier model
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)#to predict the test set
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)#print accuracy of the lists
classification_report_rf = classification_report(y_test, y_pred_rf)
print("Random Forest Classifier Classification Report:\n", classification_report_rf)
joblib.dump(rf_model,"mymodel_rf.h5")
#save the trained model
#naive_bayes
'''
NB_model = GaussianNB()
NB_model.fit(x_train, y_train)
y_pred_NB = NB_model.predict(x_test)
accuracy_NB = accuracy_score(y_test, y_pred_NB)
print("GaussianNB Accuracy:", accuracy_NB)
classification_report_NB = classification_report(y_test, y_pred_NB)
print("GaussianNB Classification Report:\n", classification_report_NB)
joblib.dump(NB_model,"mymodel_na.h5")
'''
#svm(support vector machine)--handle both linear and non-linear tasks
'''
svm_model = SVC()
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("svm Accuracy:", accuracy_svm)
classification_report_svm = classification_report(y_test, y_pred_svm)
print("svm Classification Report:\n", classification_report_svm)
joblib.dump(svm_model,"mymodel_svm.h5")
#load the saved model
loaded_rf_model=joblib.load("mymodel_rf.h5")
#predict  on new data
new_data = x_test.iloc[:5]
svm_predictions = loaded_rf_model.predict(new_data)
print("Random Forest predictions on new data:",svm_predictions)

'''
'''#load the saved model
loaded_rf_model=joblib.load("mymodel_rf.h5")
#predict  on new data
new_data = x_test.iloc[:5]
rf_predictions = loaded_rf_model.predict(new_data)
print("Random Forest predictions on new data:",rf_predictions)'''






