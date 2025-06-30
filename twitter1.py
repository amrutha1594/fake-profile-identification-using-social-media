import pandas as pd
import joblib
model = joblib.load("mymodel_ran.h5")
feature_names = model.feature_names_in_
a = int(input("Enter UserID: "))
b = str(input("Enter name: "))
c = int(input("Enter No Of Abuse Report: "))
d = int(input("Enter No Of Rejected Friend Requests: "))
e = int(input("Enter No Of Friend Thar Are Not Accepted: "))
f = int(input("Enter No Of Friends: "))
g = int(input("Enter No Of Followers: "))
h = int(input("Enter No of Likes To Unknown Account:"))
i = int(input("Enter No Of comments Per Day"))
new_data ={
    "UserID":a,
    "name":b,
    "No Of Abuse Report": c,
    "No Of Rejected Friend Requests": d,
    "No Of Friend Thar Are Not Accepted": e,
    "No Of Friends": f,
    "No Of Followers": g,
    "No of Likes To Unknown Account": h,
    "No Of comments Per Day": i
    }
new_data_df = pd.DataFrame([new_data])
new_data_encoded = new_data_df.reindex(columns=feature_names, fill_value=0)
prediction = model.predict(new_data_encoded)
if prediction==1:
    print("Original")
else:
    print("fake")
