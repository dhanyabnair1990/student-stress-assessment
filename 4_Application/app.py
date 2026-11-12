"""
Step 6: Student Wellbeing Questionnaire Prototype

Run from the project root:
    python -m streamlit run 4_Application/app.py

This prototype uses the saved Random Forest pipeline from Step 4.
It does not save questionnaire responses.
It is not a diagnostic or clinical assessment tool.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Locate the project folder and trained model.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "3_ML_Model"
    / "Model_Outputs"
    / "Random_Forest.joblib"
)

QUESTION_OPTIONS = {
    "Q1_class": ["Class 9", "Class 10"],
    "Q2_age_group": [
        "13 or below", "14", "15", "16", "17 or above", "Prefer not to say"
    ],
    "Q3_study_time_outside_school": [
        "Less than 1 hour", "1–2 hours", "2–3 hours",
        "More than 3 hours", "It varies"
    ],
    "Q4_schoolwork_manageable": [
        "Strongly agree", "Agree", "Neither agree nor disagree",
        "Disagree", "Strongly disagree"
    ],
    "Q5_think_about_tests": [
        "Never", "Rarely", "Sometimes", "Often", "Very often"
    ],
    "Q6_confident_with_academic_work": [
        "Strongly agree", "Agree", "Neither agree nor disagree",
        "Disagree", "Strongly disagree"
    ],
    "Q7_compare_with_classmates": [
        "Never", "Rarely", "Sometimes", "Often", "Very often"
    ],
    "Q8_difficulty_concentrating": [
        "Never", "Rarely", "Sometimes", "Often", "Very often"
    ],
    "Q9_comfortable_discussing_concerns": [
        "Strongly agree", "Agree", "Neither agree nor disagree",
        "Disagree", "Strongly disagree"
    ],
    "Q10_family_expectations_influence": [
        "Strongly agree", "Agree", "Neither agree nor disagree",
        "Disagree", "Strongly disagree"
    ],
    "Q11_sleep_on_school_nights": [
        "Less than 5 hours", "5–6 hours", "6–7 hours",
        "7–8 hours", "More than 8 hours"
    ],
    "Q12_screen_time_outside_schoolwork": [
        "Less than 1 hour", "1–2 hours", "2–3 hours",
        "3–4 hours", "More than 4 hours"
    ],
    "Q13_physical_activity_days": [
        "0 days", "1–2 days", "3–4 days", "5–7 days"
    ],
    "Q14_mentally_tired_or_overwhelmed": [
        "Never", "Rarely", "Sometimes", "Often", "Very often"
    ],
}

QUESTION_LABELS = {
    "Q1_class": "I am currently studying in:",
    "Q2_age_group": "My age group is:",
    "Q3_study_time_outside_school": (
        "I spend this much time studying outside school:"
    ),
    "Q4_schoolwork_manageable": "I find my schoolwork manageable.",
    "Q5_think_about_tests": "I think about tests or exams this often:",
    "Q6_confident_with_academic_work": "I feel confident about my academic work.",
    "Q7_compare_with_classmates": (
        "I compare my academic performance with my classmates this often:"
    ),
    "Q8_difficulty_concentrating": (
        "I find it difficult to concentrate on schoolwork this often:"
    ),
    "Q9_comfortable_discussing_concerns": (
        "I feel comfortable discussing my concerns with someone."
    ),
    "Q10_family_expectations_influence": (
        "Family expectations influence the pressure I feel about schoolwork."
    ),
    "Q11_sleep_on_school_nights": (
        "I usually sleep this many hours on school nights:"
    ),
    "Q12_screen_time_outside_schoolwork": (
        "I spend this much time on screens outside schoolwork each day:"
    ),
    "Q13_physical_activity_days": (
        "I am physically active this many days in a typical week:"
    ),
    "Q14_mentally_tired_or_overwhelmed": (
        "I feel mentally tired or overwhelmed this often:"
    ),
}

st.set_page_config(
    page_title="Student Wellbeing Questionnaire",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 Student Wellbeing Questionnaire")
st.write(
    "This prototype explores students' school experiences and general wellbeing. "
    "Answer the questions below to receive an experimental questionnaire-based estimate."
)

st.info(
    "Privacy note: This prototype does not save your answers. "
    "For a real school deployment, obtain appropriate parent/guardian consent "
    "and student assent, restrict access, and establish a safeguarding process."
)

st.warning(
    "Important: This is an educational project prototype, not a medical or "
    "psychological diagnosis. Its model was trained on synthetic data and has "
    "not been validated for real students. Do not use its output to make "
    "decisions about a student's health, safety, or support eligibility."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained Random Forest model was not found.\n\n"
        "Expected location: 3_ML_Model/Model_Outputs/Random_Forest.joblib\n\n"
        "Please check that Step 4 model training completed and that the "
        "Model_Outputs folder is in the expected location."
    )
    st.stop()


@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


with st.form("student_questionnaire"):
    st.subheader("About you")
    responses = {}

    for key in ["Q1_class", "Q2_age_group"]:
        responses[key] = st.selectbox(
            QUESTION_LABELS[key],
            QUESTION_OPTIONS[key],
            index=None,
            placeholder="Select your answer",
            key=key,
        )

    st.subheader("School and study experiences")
    for key in [
        "Q3_study_time_outside_school",
        "Q4_schoolwork_manageable",
        "Q5_think_about_tests",
        "Q6_confident_with_academic_work",
        "Q7_compare_with_classmates",
        "Q8_difficulty_concentrating",
    ]:
        responses[key] = st.selectbox(
            QUESTION_LABELS[key],
            QUESTION_OPTIONS[key],
            index=None,
            placeholder="Select your answer",
            key=key,
        )

    st.subheader("Support and daily wellbeing")
    for key in [
        "Q9_comfortable_discussing_concerns",
        "Q10_family_expectations_influence",
        "Q11_sleep_on_school_nights",
        "Q12_screen_time_outside_schoolwork",
        "Q13_physical_activity_days",
        "Q14_mentally_tired_or_overwhelmed",
    ]:
        responses[key] = st.selectbox(
            QUESTION_LABELS[key],
            QUESTION_OPTIONS[key],
            index=None,
            placeholder="Select your answer",
            key=key,
        )

    submitted = st.form_submit_button("Submit answers")


if submitted:
    unanswered = [key for key, value in responses.items() if value is None]
    if unanswered:
        st.warning(
            f"Please answer all questions before submitting. "
            f"You still have {len(unanswered)} question(s) to complete."
        )
    else:
        input_df = pd.DataFrame([responses])

        try:
            predicted_category = str(model.predict(input_df)[0]).strip().title()

            st.subheader("Your questionnaire result")
            st.write(
                "The model's experimental category estimate based on these answers is:"
            )

            if predicted_category == "High":
                st.error("High")
            elif predicted_category == "Moderate":
                st.warning("Moderate")
            elif predicted_category == "Low":
                st.success("Low")
            else:
                st.info(f"Experimental category: {predicted_category}")

            st.caption(
                "This category is generated by a model trained on synthetic data. "
                "It is not a reliable measure of an individual student's stress level."
            )

            st.subheader("A little reminder for you 🌱")

            if predicted_category == "Low":
                st.markdown("**You're doing okay!**")
                st.write(
                    "Keep making time for the things you enjoy. Remember to balance "
                    "your studies with rest, hobbies and time with friends."
                )
            elif predicted_category == "Moderate":
                st.markdown("**It's okay to have challenging days.**")
                st.write(
                    "Take things one step at a time. Give yourself breaks, get enough "
                    "rest and talk to someone you trust whenever you feel overwhelmed."
                )
            else:
                st.markdown("**You don't have to figure everything out alone.**")
                st.write(
                    "School and everyday expectations can sometimes feel overwhelming. "
                    "Consider sharing how you feel with a parent, teacher or school "
                    "counsellor. Asking for support is always okay."
                )

            st.caption(
                "Remember, your result doesn't define you. You deserve support on "
                "every kind of day."
            )

        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
            st.write(
                "Please share a screenshot of this error. Do not change the model "
                "or dataset files yet."
            )
st.divider()
st.caption(
    "Student Stress Assessment Capstone | Prototype only | "
    "No answers are stored by this app."
)
