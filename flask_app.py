from flask import Flask, render_template, request
import pandas as pd
from src.mlproject.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
pipeline = PredictionPipeline()  # initialize the pipeline once

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 1. Get form data
        data =  {
            "continent": request.form["continent"],
            "education_of_employee": request.form["education_of_employee"],
            "has_job_experience": request.form["has_job_experience"],
            "requires_job_training": request.form["requires_job_training"],
            "no_of_employees": int(request.form["no_of_employees"]),
            "yr_of_estab": int(request.form["yr_of_estab"]),
            "region_of_employment": request.form["region_of_employment"],
            "prevailing_wage": float(request.form["prevailing_wage"]),
            "unit_of_wage": request.form["unit_of_wage"],
            "full_time_position": request.form["full_time_position"]
        }

        # 2. Get prediction
        prediction = pipeline.predict(data)[0]  # get first element if it's an array
        # Map encoded prediction back to string label
        label_map = {0: "Certified", 1: "Denied"}
        prediction = label_map[int(prediction)]


        # 3. Return result in template
        return render_template("home.html", prediction=f"Predicted Result: {prediction}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)