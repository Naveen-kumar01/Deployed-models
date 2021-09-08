from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import os
import pickle
import pandas as pd
from Label import MultiColumnLabelEncoder

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)

# prediction function
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


def ValuePredictor(to_predict_list):
	loaded_model = pickle.load(open("hyper1_xgboost_model.pkl", "rb"))
	result = loaded_model.predict(to_predict_list)
	return result[0]

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		pred_list = request.form.to_dict()
		pred_list['age'] = float(pred_list['age'])
		pred_list['education_num'] = float(pred_list['education_num'])
		pred_list['hours_per_week'] = float(pred_list['hours_per_week'])
		sample_input = pd.DataFrame(pd.Series(pred_list)).T
		print(sample_input)
		result = ValuePredictor(sample_input)
		if result == 0:
			salary = 'less than 50K'
		else:
			salary = 'greater than 50K'
	return render_template("result.html", prediction = salary)


if __name__ == '__main__':
	MultiColumnLabelEncoder()
	app.run(debug=True)

