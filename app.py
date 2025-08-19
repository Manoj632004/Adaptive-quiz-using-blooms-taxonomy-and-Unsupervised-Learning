from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.feature_extraction.text import TfidfVectorizer
import json, random, time
import joblib
import numpy as np
from tensorflow.keras import layers, models
import jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key' 

LIBRARY_FILE = "library.json"
DURATION_SECONDS = 20 * 60 
PREFERENCES = None

with open(LIBRARY_FILE, "r", encoding="utf-8") as f:
    QUESTIONS = json.load(f)["questions"]

pipeline = joblib.load("btl_model.pkl") 

Q_BY_ID = {q["q_id"]: q for q in QUESTIONS}

def empty_btl_scores():
    return {str(i): {"correct": 0, "total": 0} for i in range(1, 7)}

def eval_answer(q, user_ans):
    if q.get("correct_option_index") is not None:
        if user_ans is None or not str(user_ans).isdigit():
            return False
        return int(user_ans) == int(q["correct_option_index"])
    correct_text = (q.get("answer") or "").strip().lower()
    if not correct_text:
        return False
    return (user_ans or "").strip().lower() == correct_text

def train_autoencoder(prefs, questions):
    pref_vec = np.array([[prefs["remembering"], prefs["understanding"], prefs["applying"],
                   prefs["analyzing"], prefs["evaluating"], prefs["creating"]]], dtype=np.float32)
    
    texts = [f"{q['topic']} {q['question']}" for q in questions]

    vectorizer = TfidfVectorizer(max_features=500)
    X_text = vectorizer.fit_transform(texts).toarray()

    X = np.hstack([X_text, np.repeat(pref_vec, len(X_text), axis=0)])

    input_dim = X.shape[1]
    encoding_dim = 64

    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(X, X, epochs=100, batch_size=1, verbose=0)

    encoder = models.Model(input_layer, encoded)
    return autoencoder, encoder, vectorizer

def compute_difficulty(autoencoder, vectorizer, prefs, questions):
    difficulties = []
    pref_vec = np.array([[prefs["remembering"], prefs["understanding"], prefs["applying"],
                          prefs["analyzing"], prefs["evaluating"], prefs["creating"]]], dtype=np.float32)

    for q in questions:
        text = f"{q['topic']} {q['question']}"
        vec = vectorizer.transform([text]).toarray()

        x = np.hstack([vec, pref_vec])

        recon = autoencoder.predict(x, verbose=0)
        err = np.mean(np.square(x - recon))
        difficulties.append((q, err))

    return difficulties

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        topic = request.form.get('Topic').strip()
        question = request.form.get('question').strip()
        correct_answer = request.form.get('option0').strip()
        option1 = request.form.get('option1').strip()
        option2 = request.form.get('option2').strip()
        option3 = request.form.get('option3').strip()
        difficulty = request.form.get('difficulty').strip()

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
            "topic": topic,
            "question": question,
            "options": options,
            "correct_option_index": 0,
            "predicted_btl": btl,
            "difficulty": difficulty
        })

        with open(LIBRARY_FILE, "w") as f:
            json.dump(data, f, indent=4)

        flash("Question added and classified successfully!", "success")
        return redirect(url_for('index'))

    return render_template('contribute.html')

@app.route('/set_preference', methods=['GET', 'POST'])
def set_preference():
    global PREFERENCES
    if request.method == 'POST':
        # Save user preferences
        prefs = {
            "remembering": int(request.form.get("remembering")),
            "understanding": int(request.form.get("understanding")),
            "applying": int(request.form.get("applying")),
            "analyzing": int(request.form.get("analyzing")),
            "evaluating": int(request.form.get("evaluating")),
            "creating": int(request.form.get("creating"))
        }
        session['preferences'] = prefs   # store in session
        PREFERENCES = prefs     
        session.pop("autoencoder_trained", None)# reset training when prefs change
        return redirect(url_for('index'))
    return render_template('set_preference.html')


@app.route("/train_progress")
def train_progress():
    """AJAX poll route for showing training progress."""
    # In this mock version just return 100% instantly
    return jsonify({"progress": 100})

@app.route("/start_quiz")
def start_quiz():
    prefs = session.get("preferences", None)
    filtered_questions = QUESTIONS

    if prefs:  # if preferences exist, train once
        if "autoencoder_trained" not in session:
            autoencoder, _, vectorizer = train_autoencoder(prefs, QUESTIONS)
            scored = compute_difficulty(autoencoder, vectorizer, prefs, QUESTIONS)

            threshold = np.median([e for (_, e) in scored])
            filtered_questions = [q for q, e in scored if e <= threshold]

            session["autoencoder_trained"] = True
            session["filtered_ids"] = [q["q_id"] for q in filtered_questions]
        else:
            filtered_ids = session.get("filtered_ids", [])
            filtered_questions = [q for q in QUESTIONS if q["q_id"] in filtered_ids]

    # Now randomly select from filtered
    selected = random.sample(filtered_questions, min(25, len(filtered_questions)))
    order_ids = ",".join([q["q_id"] for q in selected])
    started_at = int(time.time())

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
    category_scores = request.args["category_scores"] 
    questions_order = request.args["questions_order"]
    started_at = int(request.args["started_at"])
    duration = int(request.args["duration"])

    order_list = questions_order.split(",")
    q_id = order_list[q_index]
    question_obj = Q_BY_ID[q_id]

    now = int(time.time())
    remaining_time = max(0, duration - (now - started_at))

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

    now = int(time.time())
    remaining_time = max(0, duration - (now - started_at))

    if remaining_time <= 0:
        return redirect(url_for(
            "quiz_result",
            correct_count=correct_count,
            total_count=total_count,
            category_scores=json.dumps(category_scores),
            timed_out=1
        ))

    user_ans = request.form.get("answer")
    btl_key = str(question_obj.get("predicted_btl", "Unknown"))

    total_count += 1
    if btl_key not in category_scores:
        category_scores[btl_key] = {"correct": 0, "total": 0}

    if eval_answer(question_obj, user_ans):
        correct_count += 1
        category_scores[btl_key]["correct"] += 1
    category_scores[btl_key]["total"] += 1

    q_index += 1
    if q_index >= len(order_list):
        return redirect(url_for(
            "quiz_result",
            correct_count=correct_count,
            total_count=total_count,
            category_scores=json.dumps(category_scores),
            timed_out=0
        ))

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

    percentages = {}
    for btl, s in category_scores.items():
        if isinstance(s, dict): 
            tot = max(1, s.get("total", 0))
            percentages[btl] = round(100.0 * s.get("correct", 0) / tot, 2)
        else:  
            percentages[btl] = s

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
