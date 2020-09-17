from flask import Flask,render_template,url_for,request
import pickle
import joblib


clf = pickle.load(open('new_xgb_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = clf.predict(data)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()