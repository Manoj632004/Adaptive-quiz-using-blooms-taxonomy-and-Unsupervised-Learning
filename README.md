# BTL_Questionnaire_On_CS

## Overview
This project implements an intelligent quiz system that personalizes question delivery based on both Bloom’s Taxonomy Levels (BTL) and the learner’s difficulty preferences. The system dynamically filters and recommends questions using an unsupervised learning approach (Autoencoder-based reconstruction error) to approximate difficulty levels.
By combining educational theory with machine learning, this project demonstrates a practical system for adaptive learning. It can be extended to real-world educational platforms to enhance engagement, fairness, and personalization in assessments.

## Motivation 
Traditional quizzes and assessments often follow a one-size-fits-all approach. 
- Learners face the same set of questions regardless of their skill level, learning goals, or cognitive strengths.
- Difficulty is usually predefined by instructors, which introduces bias and lacks adaptability.

This creates a mismatch between a learner’s actual ability and the questions they receive, which can lead to disengagement, and lack of challenge.

The quiz system aims to:
- Personalize assessments by aligning questions with each learner’s comfort level across Bloom’s Taxonomy.
- Automate difficulty estimation using unsupervised machine learning (autoencoders), reducing reliance on manual labeling.
- Promote adaptive learning — ensuring learners are neither under-challenged nor overwhelmed
- Lay groundwork for scalable educational tools that could be integrated into e-learning platforms, tutoring systems, or exam preparation apps.

## Approach

### 1. Data Representation
Each question is stored in a format containing:
- topic → Subject area .
- question → The actual text of the question.
- options → Multiple-choice options.
- correct_option_index → Ground-truth answer.
- predicted_btl → Cognitive category (1–6) according to Bloom’s Taxonomy.

For training, the question text and topic are concatenated and converted into vectors using TF-IDF vectorization.

Learner preferences (self reported comfort levels across BTL levels) are represented as a 6-dimensional numeric vector.

### 2. Autoencoder for Difficulty Estimation
An autoencoder is trained on the combination of:
- Question vector (semantic content).
- Learner preference vector (cognitive comfort profile).

The autoencoder attempts to reconstruct the input.

Reconstruction error acts as a proxy for difficulty:
- Low error → The question fits well with the learner’s profile (easier).
- High error → The question is outside the learner’s comfort zone (harder).

### 3. Personalized Question Filtering

- The questions is vectorized and passed through the trained autoencoder.
- Reconstruction error is computed for each question.
- Questions are filtered by median of predicted scores among all questions. Then keep only those questions whose score is less than or equal to the median.

### 4. Scoring & Feedback
The quiz results a -
- Category-wise accuracy per Bloom’s Taxonomy level.
- Overall performance percentage

This feedback loop helps learners:
- Understand where they stand.
- Identify weak areas requiring more practice.
- Progressively attempt higher-order questions when ready.

## Evaluation 

## Improvements to be made
