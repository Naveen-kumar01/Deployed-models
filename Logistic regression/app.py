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
	loaded_model = pickle.load(open("Logistic_model.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		pred_list = request.form.to_dict()
		pred_list['rate_marriage'] = float(pred_list['rate_marriage'])
		pred_list['occupation'] = float(pred_list['occupation'])
		pred_list['yrs_married'] = float(pred_list['yrs_married'])
		pred_list['children'] = float(pred_list['children'])
		pred_list['religious'] = float(pred_list['religious'])
		pred_list['educ'] = float(pred_list['educ'])
		pred_list['occupation'] = float(pred_list['occupation'])
		pred_list['occupation_husb'] = float(pred_list['occupation_husb'])
		sample_input = pd.DataFrame(pd.Series(pred_list)).T
		result = ValuePredictor(sample_input)
		if result == 1:
			affair = 'Affair is present'
			print(affair)
		else:
			affair = 'No Affair'
			print(affair)
	return render_template("result.html", prediction = affair)


if __name__ == '__main__':
    app.run()

