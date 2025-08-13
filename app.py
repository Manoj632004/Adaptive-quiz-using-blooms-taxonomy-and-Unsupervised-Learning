from flask import Flask, render_template, request, redirect, url_for, flash
import json, random, time, urllib.parse
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

LIBRARY_FILE = "library.json"
DURATION_SECONDS = 20 * 60  # 20 minutes

# Load question bank
with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)["questions"]

pipeline = joblib.load("btl_model.pkl") 

# Quick lookup by q_id
Q_BY_ID = {q["q_id"]: q for q in QUESTIONS}

def empty_btl_scores():
    # BTL 1..6; use strings as keys for consistency
    return {str(i): {"correct": 0, "total": 0} for i in range(1, 7)}

def eval_answer(q, user_ans):
    """
    Supports MCQ via correct_option_index, and text via 'answer' (optional).
    Returns True/False.
    """
    if q.get("correct_option_index") is not None:
        # user_ans is expected to be an index (string)
        if user_ans is None or not str(user_ans).isdigit():
            return False
        return int(user_ans) == int(q["correct_option_index"])
    # Fallback to text answer if provided
    correct_text = (q.get("answer") or "").strip().lower()
    if not correct_text:
        return False
    return (user_ans or "").strip().lower() == correct_text

@app.route("/")
def index():
    return render_template("index.html")

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

        with open(LIBRARY_FILE, "r") as f:
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

        with open(LIBRARY_FILE, "w") as f:
            json.dump(data, f, indent=4)

        flash("Question added and classified successfully!", "success")
        return redirect(url_for('index'))

    return render_template('contribute.html')

@app.route("/start_quiz")
def start_quiz():
    # Pick 25 random questions; store ONLY their q_ids (stateless)
    selected = random.sample(QUESTIONS, min(25, len(QUESTIONS)))
    order_ids = ",".join([q["q_id"] for q in selected])

    started_at = int(time.time())  # epoch seconds

    return redirect(url_for(
        "quiz_question",
        q_index=0,
        correct_count=0,
        total_count=0,
        category_scores=json.dumps(empty_btl_scores()),
        questions_order=order_ids,
        started_at=started_at,
        duration=DURATION_SECONDS
    ))

@app.route("/quiz_question")
def quiz_question():
    q_index = int(request.args["q_index"])
    correct_count = int(request.args["correct_count"])
    total_count = int(request.args["total_count"])
    category_scores = request.args["category_scores"]  # JSON string
    questions_order = request.args["questions_order"]
    started_at = int(request.args["started_at"])
    duration = int(request.args["duration"])

    order_list = questions_order.split(",")
    q_id = order_list[q_index]
    question_obj = Q_BY_ID[q_id]

    # Backend-enforced remaining time
    now = int(time.time())
    remaining_time = max(0, duration - (now - started_at))

    # If time already up, finish immediately
    if remaining_time <= 0:
        return redirect(url_for(
            "quiz_result",
            correct_count=correct_count,
            total_count=total_count,
            category_scores=category_scores,
            timed_out=1
        ))

    is_last = (q_index == len(order_list) - 1)

    return render_template(
        "quiz_question.html",
        question=question_obj,
        q_index=q_index,
        correct_count=correct_count,
        total_count=total_count,
        category_scores=category_scores,
        questions_order=questions_order,
        started_at=started_at,
        duration=duration,
        remaining_time=remaining_time,
        is_last=is_last
    )

@app.route("/take_quiz", methods=["POST"])
def take_quiz():
    # Read incoming fields
    q_index = int(request.form["q_index"])
    correct_count = int(request.form["correct_count"])
    total_count = int(request.form["total_count"])
    category_scores = json.loads(request.form["category_scores"])
    questions_order = request.form["questions_order"]
    started_at = int(request.form["started_at"])
    duration = int(request.form["duration"])

    order_list = questions_order.split(",")
    q_id = order_list[q_index]
    question_obj = Q_BY_ID[q_id]

    # Enforce time on backend
    now = int(time.time())
    remaining_time = max(0, duration - (now - started_at))

    # If time up, skip evaluation and finish
    if remaining_time <= 0:
        return redirect(url_for(
            "quiz_result",
            correct_count=correct_count,
            total_count=total_count,
            category_scores=json.dumps(category_scores),
            timed_out=1
        ))

    # Evaluate this question
    user_ans = request.form.get("answer")
    btl_key = str(question_obj.get("predicted_btl", "Unknown"))

    total_count += 1
    # Ensure key exists even if unexpected BTL appears
    if btl_key not in category_scores:
        category_scores[btl_key] = {"correct": 0, "total": 0}

    if eval_answer(question_obj, user_ans):
        correct_count += 1
        category_scores[btl_key]["correct"] += 1
    category_scores[btl_key]["total"] += 1

    # Next or Finish
    q_index += 1
    if q_index >= len(order_list):
        return redirect(url_for(
            "quiz_result",
            correct_count=correct_count,
            total_count=total_count,
            category_scores=json.dumps(category_scores),
            timed_out=0
        ))

    # Continue to next question; keep same started_at/duration (no reset)
    return redirect(url_for(
        "quiz_question",
        q_index=q_index,
        correct_count=correct_count,
        total_count=total_count,
        category_scores=json.dumps(category_scores),
        questions_order=questions_order,
        started_at=started_at,
        duration=duration
    ))

@app.route("/quiz_result")
def quiz_result():
    correct_count = int(request.args["correct_count"])
    total_count = int(request.args["total_count"])
    category_scores = json.loads(request.args["category_scores"])
    timed_out = int(request.args.get("timed_out", 0))

    # Compute percentages for convenience
    percentages = {}
    for btl, s in category_scores.items():
        tot = max(1, s.get("total", 0))
        percentages[btl] = round(100.0 * s.get("correct", 0) / tot, 2)

    return render_template(
        "quiz_result.html",
        correct_count=correct_count,
        total_count=total_count,
        category_scores=category_scores,
        percentages=percentages,
        timed_out=timed_out
    )

if __name__ == "__main__":
    app.run(debug=True)
