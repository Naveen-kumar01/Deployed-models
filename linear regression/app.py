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
	loaded_model = pickle.load(open("Ridge_model.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		pred_list = request.form.to_dict()
		pred_list['crim'] = float(pred_list['crim'])
		pred_list['zn'] = float(pred_list['zn'])
		pred_list['indus'] = float(pred_list['indus'])
		pred_list['chas'] = float(pred_list['chas'])
		pred_list['nox'] = float(pred_list['nox'])
		pred_list['rm'] = float(pred_list['rm'])
		pred_list['age'] = float(pred_list['age'])
		pred_list['dis'] = float(pred_list['dis'])
		pred_list['rad'] = float(pred_list['rad'])
		pred_list['tax'] = float(pred_list['tax'])
		pred_list['ptratio'] = float(pred_list['ptratio'])
		pred_list['black'] = float(pred_list['black'])
		pred_list['lstat'] = float(pred_list['lstat'])
		sample_input = pd.DataFrame(pd.Series(pred_list)).T
		result = ValuePredictor(sample_input)
	return render_template("result.html", prediction = result)


if __name__ == '__main__':
    app.run()

