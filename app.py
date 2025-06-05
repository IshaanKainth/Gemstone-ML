from flask import Flask,render_template,jsonify,request
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/form',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            cut=str(request.form.get('cut')),
            color=str(request.form.get('color')),
            clarity=str(request.form.get('clarity')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z'))
        )
        final_df=data.get_data_as_df()
        predict_pipeline=PredictionPipeline()

        pred=predict_pipeline.prediction(final_df)

        return render_template('submit.html',prediction=round(pred[0],2))


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

