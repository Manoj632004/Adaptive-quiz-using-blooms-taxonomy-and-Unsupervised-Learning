import streamlit as st
import json
import os
import joblib
import pandas as pd

pipeline = joblib.load("btl_model.pkl")

QUESTIONS_JSON = "library.json"

if not os.path.exists(QUESTIONS_JSON):
    with open(QUESTIONS_JSON, "w") as f:
        json.dump([], f)

st.set_page_config(page_title="BTL Questionnaire", layout="centered")
st.title("BTL Level Classifier for Computer Science Questions")

menu = ["Home", "Contribute Question", "Question Library"]
choice = st.sidebar.radio("Navigation", menu)

if choice == "Home":
    st.subheader("Welcome!")
    st.write("This app classifies computer science questions into Bloom's Taxonomy Levels (BTL).")
    st.write("You can contribute new questions or explore the existing library.")

elif choice == "✍ Contribute Question":
    st.subheader("Contribute a Question")
    question = st.text_area("Enter your question:")

    if st.button("Classify & Save"):
        if not question.strip():
            st.error("Question cannot be empty.")
        else:
            # Load existing data
            with open(QUESTIONS_JSON, "r") as f:
                data = json.load(f)

            # Check for duplicate (case-insensitive match)
            if any(q["question"].strip().lower() == question.strip().lower() for q in data):
                st.warning("⚠ This question already exists in the library.")
            else:
                # Predict BTL
                btl = int(pipeline.predict([question])[0])

                # Add new question
                data.append({
                    "question": question.strip(),
                    "predicted_btl": btl
                })

                # Save back to JSON
                with open(QUESTIONS_JSON, "w") as f:
                    json.dump(data, f, indent=4)

                st.success(f"Question classified as **BTL Level {btl}** and saved!")

elif choice == "📂 Question Library":
    st.subheader("Question Library")
    with open(QUESTIONS_JSON, "r") as f:
        data = json.load(f)

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.info("No questions in the library yet.")
