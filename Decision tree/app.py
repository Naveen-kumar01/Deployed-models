from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import os
import numpy as np
import pickle
import pandas as pd

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)

# prediction function
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


def ValuePredictor(to_predict_list):
	loaded_model = pickle.load(open("Decision_model.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		pred_list = request.form.to_dict()
		pred_list['Pclass'] = int(pred_list['Pclass'])
		pred_list['Age'] = float(pred_list['Age'])
		pred_list['Fare'] = float(pred_list['Fare'])
		pred_list['relatives'] = int(pred_list['relatives'])
		sample_input = pd.DataFrame(pd.Series(pred_list)).T
		result = ValuePredictor(sample_input)
		if result == 1:
			r = 'Survived'
			print(r)
		else:
			r = 'Did not Survived'
			print(r)
	return render_template("result.html", prediction = r)


if __name__ == '__main__':
    app.run()

