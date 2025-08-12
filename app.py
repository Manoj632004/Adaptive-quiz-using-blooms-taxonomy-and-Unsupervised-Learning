from flask import Flask, request, redirect, render_template, url_for, flash, session
import json
import os
import joblib 
from collections import defaultdict
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

pipeline = joblib.load("btl_model.pkl") 

QUESTIONS_JSON = "library.json"

if not os.path.exists(QUESTIONS_JSON):
    with open(QUESTIONS_JSON, "w") as f:
        json.dump([], f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        question = request.form.get('question').strip()
        correct_answer = request.form.get('option0').strip()
        option1 = request.form.get('option1').strip()
        option2 = request.form.get('option2').strip()
        option3 = request.form.get('option3').strip()

        if not question:
            flash("Question cannot be empty.", "error")
            return redirect(url_for('contribute'))

        if not correct_answer:
            flash("Correct answer cannot be empty.", "error")
            return redirect(url_for('contribute'))

        options = [correct_answer]
        if option1:
            options.append(option1)
        if option2:
            options.append(option2)
        if option3:
            options.append(option3)

        btl = int(pipeline.predict([question])[0])

        with open(QUESTIONS_JSON, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {"questions": data}
            except Exception:
                data = {"questions": []}

        if "questions" not in data:
            data["questions"] = []

        q_id = f"q_{len(data['questions']) + 1}"

        data["questions"].append({
            "q_id": q_id,
            "question": question,
            "options": options,
            "correct_option_index": 0,
            "predicted_btl": btl
        })

        with open(QUESTIONS_JSON, "w") as f:
            json.dump(data, f, indent=4)

        flash("Question added and classified successfully!", "success")
        return redirect(url_for('home'))

    return render_template('contribute.html')

def load_questions():
    with open(QUESTIONS_JSON, "r") as f:
        return json.load(f)["questions"]

@app.route("/start_quiz")
def start_quiz():
    questions = load_questions()
    quiz_questions = random.sample(questions, min(25, len(questions)))
    session["quiz_questions"] = quiz_questions
    session["current_index"] = 0
    session["answers"] = {}
    return redirect(url_for("take_quiz"))

@app.route("/take_quiz", methods=["GET", "POST"])
def take_quiz():
    if "quiz_questions" not in session:
        return redirect(url_for("home"))#throw some sort of message in home screen

    if request.method == "POST":
        selected_answer = request.form.get("answer")
        q_index = session["current_index"]
        if selected_answer is not None:
            session["answers"][q_index] = selected_answer

        session["current_index"] += 1
        if session["current_index"] >= len(session["quiz_questions"]):
            return redirect(url_for("quiz_result"))
    q_index = session["current_index"]
    question = session["quiz_questions"][q_index]
    total_questions = len(session["quiz_questions"])
    is_last = (q_index == total_questions - 1)

    return render_template(
        "take_quiz.html",
        question=question,
        q_number=q_index + 1,
        total=total_questions,
        is_last=is_last
    )

@app.route("/quiz_result")
def quiz_result():
    questions = session.get("quiz_questions", [])
    answers = session.get("answers", {})

    category_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for idx, q in enumerate(questions):
        category = q.get("predicted_btl", "Unknown")
        category_results[category]["total"] += 1
        correct_index = q.get("correct_option_index")
        if correct_index is not None and str(correct_index) == answers.get(idx):
            category_results[category]["correct"] += 1

    return render_template("quiz_result.html", results=category_results)

if __name__ == '__main__':
    app.run(debug=True)
