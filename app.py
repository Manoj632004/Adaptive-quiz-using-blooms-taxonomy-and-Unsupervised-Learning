from flask import Flask, request, redirect, render_template, url_for, flash
import json
import os
import joblib  # for loading model
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Load the trained model pipeline
pipeline = joblib.load("btl_model.pkl")  # Use your saved model here

QUESTIONS_JSON = "library.json"

# Ensure JSON file exists
if not os.path.exists(QUESTIONS_JSON):
    with open(QUESTIONS_JSON, "w") as f:
        json.dump([], f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        question = request.form.get('question')

        if not question.strip():
            flash("Question cannot be empty.", "error")
            return redirect(url_for('contribute'))

        # Run prediction
        btl = int(pipeline.predict([question])[0])

        # Add to JSON
        with open(QUESTIONS_JSON, "r") as f:
            data = json.load(f)

        data.append({
            "question": question,
            "predicted_btl": btl
        })

        with open(QUESTIONS_JSON, "w") as f:
            json.dump(data, f, indent=4)

        flash("Question added and classified successfully!", "success")
        return redirect(url_for('home'))

    return render_template('contribute.html')
if __name__ == '__main__':
    app.run(debug=True)
