import pandas as pd
import joblib

# Load the model
model = joblib.load("mymodel_ran.h5")  # Assuming it's saved as a .pkl file

# Get feature names from the model
feature_names = model.feature_names_in_

# Collect user input
a = int(input("Enter UserID: "))
b = input("Enter name: ")  # Assuming 'name' is a string
c = int(input("Enter No Of Abuse Report: "))
d = int(input("Enter No Of Rejected Friend Requests: "))
e = int(input("Enter No Of Friend That Are Not Accepted: "))  # Corrected typo 'Thar' to 'That'
f = int(input("Enter No Of Friends: "))
g = int(input("Enter No Of Followers: "))
h = int(input("Enter No of Likes To Unknown Account:"))
i = int(input("Enter No Of comments Per Day: "))

# Create a dictionary with user input
new_data ={
    "UserID": a,
    "name": b,
    "No Of Abuse Report": c,
    "No Of Rejected Friend Requests": d,
    "No Of Friend That Are Not Accepted": e,
    "No Of Friends": f,
    "No Of Followers": g,
    "No of Likes To Unknown Account": h,
    "No Of comments Per Day": i
}

# Create a DataFrame from the dictionary
new_data_df = pd.DataFrame([new_data])

# Reindex the DataFrame to match the model's feature names and fill with 0
new_data_encoded = new_data_df.reindex(columns=feature_names, fill_value=0)

# Make prediction using the loaded model
prediction = model.predict(new_data_encoded)

# Print prediction
print(prediction)

# Output result based on prediction
if prediction[0] == 0:  # Assuming prediction[0] is the first prediction in the array
    print("Original")
else:
    print("Fake")
