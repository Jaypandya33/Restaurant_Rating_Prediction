from src.mlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("index.html")
    
    else:
        data=CustomData(
            
            online_order = str(request.form.get('online_order')),
            book_table =str(request.form.get('book_table')),
            votes = float(request.form.get('votes')),
            location = str(request.form.get('location')),
            rest_type = str(request.form.get('rest_type')),
            cuisines = str(request.form.get('cuisines')),
            cost_for_two_people = float(request.form.get('cost_for_two_people')),
            type = str(request.form.get('type'))
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        pred=round(pred[0],2)
        
        return render_template("result.html",prediction = str(pred))

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)