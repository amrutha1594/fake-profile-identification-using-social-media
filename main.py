from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/add', methods=['POST'])
def add():
    try:
        import pandas as pd
        import joblib
        model = joblib.load("mymodel_rf.h5")
        feature_names = model.feature_names_in_
        a = int(request.form['a'])
        b = float(request.form['b'])
        c = int(request.form['c'])
        d = float(request.form['d'])
        e = int(request.form['e'])
        f = int(request.form['f'])
        g = int(request.form['g'])
        h = int(request.form['h'])
        i = int(request.form['i'])
        j = int(request.form['j'])
        k = int(request.form['k'])
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
        result = model.predict(new_data_encoded)
        


        
    except ValueError:
        result = "Invalid input! Please enter numbers only."
    
    return render_template('result.html',result="Original" if result==0 else "Fake")

@app.route('/predict')
def index():
    return render_template('index.html')
def result():
    return render_template('result.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/twitter', methods=['POST'])
def twitter():
    try:
        import pandas as pd
        import joblib
        model = joblib.load("mymodel_ran.h5")
        feature_names = model.feature_names_in_
        a = int(request.form['a'])
        b = str(request.form['b'])
        c = int(request.form['c'])
        d = int(request.form['d'])
        e = int(request.form['e'])
        f = int(request.form['f'])
        g = int(request.form['g'])
        h = int(request.form['h'])
        i = int(request.form['i'])
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
        result1 = model.predict(new_data_encoded)
        



        
    except ValueError:
        result1 = "Invalid input! Please enter numbers only."
    
    return render_template('result1.html',result1="Original" if result1==1 else "Fake")


@app.route('/predict1')
def index1():
    return render_template('index1.html')
def result1():
    return render_template('result1.html')



if __name__ == '__main__':
    app.run(debug=True,port=8080)

