"""
Student Stress Assessment & Wellbeing Support System
Run from the project root:
    python -m streamlit run 4_Application/app.py
This prototype does not save questionnaire responses.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "3_ML_Model" / "Model_Outputs" / "Random_Forest.joblib"

QUESTION_OPTIONS = {
    "Q1_class": ["Class 9", "Class 10"],
    "Q2_age_group": ["13 or below", "14", "15", "16", "17 or above", "Prefer not to say"],
    "Q3_study_time_outside_school": ["Less than 1 hour", "1–2 hours", "2–3 hours", "More than 3 hours", "It varies"],
    "Q4_schoolwork_manageable": ["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"],
    "Q5_think_about_tests": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "Q6_confident_with_academic_work": ["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"],
    "Q7_compare_with_classmates": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "Q8_difficulty_concentrating": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
    "Q9_comfortable_discussing_concerns": ["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"],
    "Q10_family_expectations_influence": ["Strongly agree", "Agree", "Neither agree nor disagree", "Disagree", "Strongly disagree"],
    "Q11_sleep_on_school_nights": ["Less than 5 hours", "5–6 hours", "6–7 hours", "7–8 hours", "More than 8 hours"],
    "Q12_screen_time_outside_schoolwork": ["Less than 1 hour", "1–2 hours", "2–3 hours", "3–4 hours", "More than 4 hours"],
    "Q13_physical_activity_days": ["0 days", "1–2 days", "3–4 days", "5–7 days"],
    "Q14_mentally_tired_or_overwhelmed": ["Never", "Rarely", "Sometimes", "Often", "Very often"],
}

QUESTION_LABELS = {
    "Q1_class": "I am currently studying in:",
    "Q2_age_group": "My age group is:",
    "Q3_study_time_outside_school": "I spend this much time studying outside school:",
    "Q4_schoolwork_manageable": "I find my schoolwork manageable.",
    "Q5_think_about_tests": "I think about tests or exams this often:",
    "Q6_confident_with_academic_work": "I feel confident about my academic work.",
    "Q7_compare_with_classmates": "I compare my academic performance with my classmates this often:",
    "Q8_difficulty_concentrating": "I find it difficult to concentrate on schoolwork this often:",
    "Q9_comfortable_discussing_concerns": "I feel comfortable discussing my concerns with someone.",
    "Q10_family_expectations_influence": "Family expectations influence the pressure I feel about schoolwork.",
    "Q11_sleep_on_school_nights": "I usually sleep this many hours on school nights:",
    "Q12_screen_time_outside_schoolwork": "I spend this much time on screens outside schoolwork each day:",
    "Q13_physical_activity_days": "I am physically active this many days in a typical week:",
    "Q14_mentally_tired_or_overwhelmed": "I feel mentally tired or overwhelmed this often:",
}

st.set_page_config(page_title="Student Wellbeing Check-in", page_icon="🌱", layout="centered")
for key, default in [("page", "home"), ("responses", None), ("prediction", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def personalised_suggestions(a):
    items = []
    # Q3: study time
    if a["Q3_study_time_outside_school"] == "More than 3 hours":
        items.append(("Make room for breaks", "If studying is taking up much of your day, remember to pause too. A short break between study sessions can help you return feeling fresher."))
    # Q4: manageability
    if a["Q4_schoolwork_manageable"] in ["Disagree", "Strongly disagree"]:
        items.append(("Take schoolwork one step at a time", "When schoolwork feels hard to manage, try choosing one small task to start with. You can also ask a teacher or classmate to explain anything that feels unclear."))
    # Q5: exam thoughts
    if a["Q5_think_about_tests"] in ["Often", "Very often"]:
        items.append(("Make exam preparation feel manageable", "If exams are on your mind a lot, try splitting revision into small tasks. You don't have to prepare for everything in one sitting."))
    # Q6: confidence
    if a["Q6_confident_with_academic_work"] in ["Disagree", "Strongly disagree"]:
        items.append(("Notice the progress you're making", "It's normal not to feel confident all the time. Try noticing one thing you understood or improved on today, even if it seems small."))
    # Q7: comparison
    if a["Q7_compare_with_classmates"] in ["Often", "Very often"]:
        items.append(("Focus on your own progress", "It's easy to compare marks or performance with friends. Everyone learns at a different pace, so try measuring progress against what you could do before."))
    # Q8: concentration
    if a["Q8_difficulty_concentrating"] in ["Often", "Very often"]:
        items.append(("Try shorter study sessions", "If focusing has been difficult, try studying for a shorter stretch, putting away distractions and taking a quick break before you begin again."))
    # Q9: talking about concerns
    if a["Q9_comfortable_discussing_concerns"] in ["Disagree", "Strongly disagree"]:
        items.append(("Find someone you feel comfortable with", "You don't have to explain everything at once. It may help to think of one trusted adult, teacher or school counsellor you could approach when you're ready."))
    # Q10: family expectations
    if a["Q10_family_expectations_influence"] in ["Agree", "Strongly agree"]:
        items.append(("Talk about the pressure you're feeling", "If expectations at home feel heavy, consider sharing what has been difficult with someone you trust. A calm conversation can be a good place to start."))
    # Q11: sleep
    if a["Q11_sleep_on_school_nights"] in ["Less than 5 hours", "5–6 hours"]:
        items.append(("Give yourself time to rest", "It sounds like your school-night sleep may be on the shorter side. If you can, try winding down a little earlier and keeping a regular bedtime. Rest matters too."))
    # Q12: screen time
    if a["Q12_screen_time_outside_schoolwork"] == "More than 4 hours":
        items.append(("Take a little screen break", "You mentioned spending quite a bit of time on screens outside schoolwork. How about taking a break from your phone for a while and doing something you enjoy offline?"))
    elif a["Q12_screen_time_outside_schoolwork"] == "3–4 hours":
        items.append(("Make space for offline time", "You could try swapping a little screen time for something away from your device, like music, a hobby or a chat with a friend."))
    # Q13: physical activity
    if a["Q13_physical_activity_days"] == "0 days":
        items.append(("Try getting outside for a bit", "If you haven't been very active lately, start small. A short walk, cycling or playing a game outdoors can be a nice change of pace."))
    elif a["Q13_physical_activity_days"] == "1–2 days":
        items.append(("Find an activity you enjoy", "If it fits into your week, you could add another day of movement. It doesn't need to be a workout; walking, dancing or playing a sport counts."))
    # Q14: tiredness
    if a["Q14_mentally_tired_or_overwhelmed"] in ["Often", "Very often"]:
        items.append(("Give yourself a moment to reset", "Sounds like you've had quite a lot on your mind. Make a little time to do something calming, and consider talking to someone you trust if it keeps feeling like too much."))
    return items[:3]

def home():
    st.title("🌱 Student Wellbeing Check-in")
    st.subheader("A little time to check in with yourself.")
    st.write("School, friends, exams and everyday life can bring all kinds of feelings. This short questionnaire is a chance to reflect on how things have been going for you. There are no right or wrong answers.")
    st.info("Your answers are used only to generate the result in this session. This prototype does not save your responses.")
    st.warning("This is an educational project prototype, not a medical or psychological assessment. Its model was trained on synthetic data and has not been validated with real students. It cannot diagnose stress or replace support from a trusted adult or qualified professional.")
    if st.button("Start check-in", type="primary", use_container_width=True):
        st.session_state.responses = None
        st.session_state.prediction = None
        st.session_state.page = "questionnaire"
        st.rerun()

def questionnaire():
    st.title("A few questions about you 🌱")
    st.write("Choose the answer that feels closest to your experience.")
    if not MODEL_PATH.exists():
        st.error("The trained Random Forest model was not found. Check 3_ML_Model/Model_Outputs/Random_Forest.joblib.")
        return
    with st.form("student_questionnaire"):
        responses = {}
        groups = [
            ("About you", ["Q1_class", "Q2_age_group"]),
            ("School and study experiences", ["Q3_study_time_outside_school", "Q4_schoolwork_manageable", "Q5_think_about_tests", "Q6_confident_with_academic_work", "Q7_compare_with_classmates", "Q8_difficulty_concentrating"]),
            ("Support and daily wellbeing", ["Q9_comfortable_discussing_concerns", "Q10_family_expectations_influence", "Q11_sleep_on_school_nights", "Q12_screen_time_outside_schoolwork", "Q13_physical_activity_days", "Q14_mentally_tired_or_overwhelmed"]),
        ]
        for heading, keys in groups:
            st.subheader(heading)
            for key in keys:
                responses[key] = st.selectbox(QUESTION_LABELS[key], QUESTION_OPTIONS[key], index=None, placeholder="Select your answer", key=f"form_{key}")
        submitted = st.form_submit_button("See my result", type="primary", use_container_width=True)
    if submitted:
        missing = [k for k, v in responses.items() if v is None]
        if missing:
            st.warning(f"You still have {len(missing)} question(s) to answer. Have a look above and submit again.")
        else:
            try:
                model = load_model(str(MODEL_PATH))
                st.session_state.responses = responses
                st.session_state.prediction = str(model.predict(pd.DataFrame([responses]))[0]).strip().title()
                st.session_state.page = "results"
                st.rerun()
            except Exception as exc:
                st.error(f"We couldn't generate a result just now: {exc}")
                st.caption("Please check the model file and try again.")
    if st.button("Back to home"):
        st.session_state.page = "home"
        st.rerun()

def results():
    a = st.session_state.responses
    category = st.session_state.prediction
    st.title("Your check-in result 🌱")
    st.write("Thanks for taking a few minutes to check in with yourself.")
    st.subheader("Your questionnaire result")
    st.write("The model's experimental category estimate is:")
    if category == "High":
        st.error("High")
    elif category == "Moderate":
        st.warning("Moderate")
    elif category == "Low":
        st.success("Low")
    else:
        st.info(f"Experimental category: {category}")
    st.caption("This is an experimental estimate from a model trained on synthetic data. It is not a reliable measure of an individual student's stress level.")
    st.subheader("A little note for you")
    if category == "Low":
        st.write("Keep making time for the things you enjoy. Studies are only one part of life, so remember to make room for rest, friends and hobbies too.")
    elif category == "Moderate":
        st.write("Some days can feel more demanding than others. Take things one step at a time, and remember that you can ask for help when you need it.")
    elif category == "High":
        st.write("If things have been feeling heavy lately, you don't have to handle everything on your own. Consider talking with a parent, teacher or school counsellor you feel comfortable with.")
    else:
        st.write("Whatever this result says, your answers are only one small snapshot of how things may be feeling right now.")
    tips = personalised_suggestions(a)
    st.subheader("A few ideas based on your answers")
    if tips:
        for title, message in tips:
            st.markdown(f"**{title}**")
            st.write(message)
    else:
        st.write("You haven't selected any answers that call for a specific suggestion from this check-in. Keep noticing what helps you feel balanced, and make time for the things you enjoy.")
    st.info("These are general, supportive ideas based on the answers you selected. They are not instructions or a diagnosis. If you are worried about how you feel, speak with a trusted adult or school counsellor.")
    st.caption("Your result does not define you. You deserve care and support on every kind of day.")
    left, right = st.columns(2)
    with left:
        if st.button("Retake check-in", type="primary", use_container_width=True):
            st.session_state.responses = None
            st.session_state.prediction = None
            st.session_state.page = "questionnaire"
            st.rerun()
    with right:
        if st.button("Back to home", use_container_width=True):
            st.session_state.responses = None
            st.session_state.prediction = None
            st.session_state.page = "home"
            st.rerun()

if st.session_state.page == "home":
    home()
elif st.session_state.page == "questionnaire":
    questionnaire()
elif st.session_state.page == "results":
    results()

st.divider()
st.caption("Student Stress Assessment Capstone | Prototype only | Questionnaire answers are not stored by this app.")
