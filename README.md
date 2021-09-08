# Deployed-models
Various models deployed on heroku,AWS,GCP,Azure.
Step 1: We will first build a model using sklearn and linear regression using banglore home prices dataset from kaggle.com. During model building we will cover almost all data science concepts such as data load and cleaning, outlier detection and removal, feature engineering, dimensionality reduction, gridsearchcv for hyperparameter tunning, k fold cross validation all of which must be included in the pipeline for the sequential execution of the process required.

Step 2: Second step would be to write a python flask server that uses the saved model to serve http requests.

Step 3: Third component is the website built in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price.

Step 4: Deployment on AWS

All the model contain pipeline implementation on the model, the user just have to enter the required field and the data 
will automatically be Scaled and Transformed using the encoding libraries present in the sklearn library and then the pipeline 
also include the training of the model and the hyperparameter tunung for reducing the risk of overfitting the data.  


The heroku platform link are given below - 

1. Random Forest model on boston house pricing with all the attributes -    https://random-forest-model.herokuapp.com/

2. Logistic regression model on Extramarital affair dataset which present in the machine learning repositories -    https://logistic-model.herokuapp.com/

3. Decision tree model on Titanic dataset from kaggle -   https://decision-tree-model.herokuapp.com/

4. Linear regression model on boston house pricing data -  https://lr-model-assignment.herokuapp.com/
                   
5.Xgboost model on Adult dataset used for salary prediction -   https://xg-model.herokuapp.com/
