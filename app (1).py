
from flask import Flask, request,render_template
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    Open = float(request.form['Open'])
    High = float(request.form['High'])
    Low = float(request.form['Low'])
    Volume = float(request.form['Volume'])
    prediction= model.predict([[Open, High, Low, Volume]])
    
    return render_template('index.html', prediction_text='The Close Prediction is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)