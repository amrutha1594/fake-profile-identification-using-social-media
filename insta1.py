import pandas as pd
import joblib
model = joblib.load("mymodel_rf.h5")
feature_names = model.feature_names_in_
a = int(input("Enter profile pic (0 or 1): "))
b = float(input("Enter nums/length username: "))
c = int(input("Enter fullname words: "))
d = float(input("Enter nums/length fullname: "))
e = int(input("Enter name == username (0 or 1): "))
f = int(input("Enter description : "))
g = int(input("Enter external URL (0 or 1): "))
h = int(input("Enter private (0 or 1): "))
i = int(input("Enter #posts: "))
j = int(input("Enter #followers: "))
k = int(input("Enter #follows: "))
new_data = {
    "profile pic": a,
    "nums/length username": b,
    "fullname words": c,
    "nums/length fullname": d,
    "name==username": e,
    "description": f,
    "external URL": g,
    "private": h,
    "#posts": i,
    "#followers": j,
    "#follows": k
}
new_data_df = pd.DataFrame([new_data])
new_data_encoded = new_data_df.reindex(columns=feature_names, fill_value=0)
prediction = model.predict(new_data_encoded)
print(prediction)
if prediction==0:
    print("Orginal")
else:
    print("Fake")

