import modelcreation
from flask import Flask, jsonify, request, render_template
from sklearn.externals import joblib
app = Flask(__name__,template_folder='template')
MODEL_FILE = 'regression_model-v4.pkl'
log_estimator = joblib.load(MODEL_FILE)

@app.route('/')
def home():
    global log_estimator
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    inputData = input("Please enter the input text::")
    result =  modelcreation.ModelCreation.scorePredictMethod(inputData)
    return render_template('index.html', prediction_text='Predicted CVE SCore is {}'.format(result))
if __name__ == '__main__':
    app.run(port=8086, debug=True)

