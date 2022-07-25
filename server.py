from flask import Flask,request,jsonify
import numpy as np
import pickle
import pandas as pd
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')
    input_query = np.array([[cgpa,iq,profile_score]])
    df = pd.DataFrame(input_query, columns=['cgpa', 'iq', 'profile_score'])
    print(model)
    result = model.predict(df)
    #result = model.predict([0,0])

    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    app.run(debug=True)