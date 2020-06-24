import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import pandas as pd

#------- new add
import io
import random
import StringIO
import base64

from flask import Flask, make_response
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
#--------
app = Flask(__name__)

#--------MODEL---------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

df = pd.read_csv('mywebscrapBrentFinal.csv')
df.columns = ['Date', 'Close', 'Open', 'High', 'Low']
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_index(ascending=False).reset_index(drop=True)
df.set_index('Date', inplace=True)
brent = df[['Close']].copy()
forecast_out = 30
brent['Prediction'] = df['Close'].shift(-forecast_out).copy()
scaler = MinMaxScaler(feature_range=(0,1))
scaled_brent = scaler.fit_transform(brent)
x = [[i[0]] for i in scaled_brent]
x = np.array(x[:-forecast_out])
y = [i[1] for i in scaled_brent]
y = np.array(y[:-forecast_out])
x_train, x_test, y_train, y_test = train_test_split(
	x,
	y,
	test_size=0.2,
    random_state=0
	)
 
df2 = pd.read_csv('wtiprice.csv')
df2.columns = ['Date', 'Close', 'Open', 'High', 'Low']
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.sort_index(ascending=False).reset_index(drop=True)
df2.set_index('Date', inplace=True)
wti = df2[['Close']].copy()
forecast_out2 = 30
wti['Prediction'] = df2['Close'].shift(-forecast_out2).copy()
scaler2 = MinMaxScaler(feature_range=(0,1))
scaled_wti = scaler2.fit_transform(wti)
x2 = [[i[0]] for i in scaled_wti]
x2 = np.array(x[:-forecast_out])
y2 = [i[1] for i in scaled_wti]
y2 = np.array(y[:-forecast_out])
x_train2, x_test2, y_train2, y_test2 = train_test_split(
	x2,
	y2,
	test_size=0.2,
    random_state=0
	)




#--------MODEL---------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    commodity_type = request.form['commodity']
    model_type = request.form['modeltype']

	#output = round(prediction[0],2)
    if commodity_type=='wtiradio' and model_type=='svm':
	   	return redirect(url_for('wtisvm'))
    elif commodity_type=='wtiradio' and model_type=='lr':
    	return redirect(url_for('wtilr'))
    	#return redirect(url_for('success',name = commodity_type))
    elif commodity_type=='brentradio' and model_type=='svm':
    	return redirect(url_for('svm'))
    elif commodity_type=='brentradio' and model_type=='lr':
    	return redirect(url_for('lr')) 
    else:
    	return render_template('index.html',prediction_text = 'Please choose the prediction option')
 
@app.route('/plot.png')
def plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    response=make_response(output.getvalue())
    response.mimetype='image/png'
    return response

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['commodity'].values()
      return render_template('index.html',prediction_text = 'Our prediction: 112')
      #return redirect(url_for('success',name = user))

#-----SVM-------
@app.route('/wtisvm')
def wtisvm():
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(x_train2, y_train2)
	svm_confidence = svr_rbf.score(x_test2, y_test2)

	x_forecast2 = np.array([[i[0]] for i in scaled_wti][-30:])
	svm_prediction = svr_rbf.predict(x_forecast2)
	svm_pred = scaled_wti.copy()
	svm_pred[-30:,1] = svm_prediction
	svm_result = pd.DataFrame(scaler.inverse_transform(svm_pred),columns=['Close', 'Prediction'])
	svm_result[-31:]

	img = io.BytesIO()
	plt.plot(wti.index[-30:],svm_prediction,color='blue', linestyle='dashed', marker='o', markerfacecolor='yellow', markersize=12)
	plt.xticks(rotation=70)
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title('WTI CRUDE OIL (SVM)')
	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	plt.close()
	return '<h1>WTI CRUDE OIL (SVM PREDICTION)</h1><img src="data:image/png;base64,{}"><br><a href="/">Home</a>'.format(plot_url)

@app.route('/wtilr')
def wtilr():
	lr = LinearRegression()
	lr.fit(x_train2, y_train2)
	lr_confidence = lr.score(x_test2, y_test2)

	x_forecast2 = np.array([[i[0]] for i in scaled_wti][-30:])
	lr_prediction = lr.predict(x_forecast2)
	lr_pred = scaled_wti.copy()
	lr_pred[-30:,1] = lr_prediction
	lr_result = pd.DataFrame(scaler.inverse_transform(lr_pred),columns=['Close', 'Prediction'])
	lr_result[-31:]

	img = io.BytesIO()
	plt.plot(wti.index[-30:],lr_prediction,color='red', linestyle='dashed', marker='o', markerfacecolor='yellow', markersize=12)
	plt.xticks(rotation=70)
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title('WTI CRUDE OIL (LR)')
	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	plt.close()
	return '<h1>WTI CRUDE OIL (LINEAR REGRESSION PREDICTION)</h1><img src="data:image/png;base64,{}"><br><a href="/">Home</a>'.format(plot_url)



#-----SVM-------
@app.route('/svm')
def svm():
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(x_train, y_train)
	svm_confidence = svr_rbf.score(x_test, y_test)

	x_forecast = np.array([[i[0]] for i in scaled_brent][-30:])
	svm_prediction = svr_rbf.predict(x_forecast)
	svm_pred = scaled_brent.copy()
	svm_pred[-30:,1] = svm_prediction
	svm_result = pd.DataFrame(scaler.inverse_transform(svm_pred),columns=['Close', 'Prediction'])
	svm_result[-31:]

	img = io.BytesIO()
	plt.plot(brent.index[-30:],svm_prediction,color='green', linestyle='dashed', marker='o', markerfacecolor='red', markersize=12)
	plt.xticks(rotation=70)
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title('BRENT CRUDE OIL (SVM)')
	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	plt.close()
	return '<h1>BRENT CRUDE OIL (SVM PREDICTION)</h1><img src="data:image/png;base64,{}"><br><a href="/">Home</a>'.format(plot_url)

#-----LR-------
@app.route('/lr')
def lr():
	lr = LinearRegression()
	lr.fit(x_train, y_train)
	lr_confidence = lr.score(x_test, y_test)

	x_forecast = np.array([[i[0]] for i in scaled_brent][-30:])
	lr_prediction = lr.predict(x_forecast)
	lr_pred = scaled_brent.copy()
	lr_pred[-30:,1] = lr_prediction
	lr_result = pd.DataFrame(scaler.inverse_transform(lr_pred),columns=['Close', 'Prediction'])
	lr_result[-31:]

	img = io.BytesIO()
	plt.plot(brent.index[-30:],lr_prediction,color='orange', linestyle='dashed', marker='o', markerfacecolor='red', markersize=12)
	plt.xticks(rotation=70)

	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title('BRENT CRUDE OIL (LR)')

	plt.savefig(img,format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	plt.close()
	return '<h1>BRENT CRUDE OIL (LINEAR REGRESSION PREDICTION)</h1><img src="data:image/png;base64,{}"><br><a href="/">Home</a>'.format(plot_url)



if __name__ == "__main__":
    app.run()
