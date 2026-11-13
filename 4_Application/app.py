"""
Student Stress Assessment & Wellbeing Support System
Run from the project root:
    python -m streamlit run 4_Application/app.py

Prototype only. Responses and contact details are held in Streamlit session
state for this demonstration and are not saved to a database or emailed.
The model was trained on synthetic data and is not a diagnostic tool.
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
    "Q1_class": "Which class are you in?",
    "Q2_age_group": "What is your age group?",
    "Q3_study_time_outside_school": "How much time do you usually spend studying outside school?",
    "Q4_schoolwork_manageable": "I find my schoolwork manageable.",
    "Q5_think_about_tests": "How often do you think about tests or exams?",
    "Q6_confident_with_academic_work": "I feel confident about my academic work.",
    "Q7_compare_with_classmates": "How often do you compare your academic performance with classmates?",
    "Q8_difficulty_concentrating": "How often do you find it difficult to concentrate on schoolwork?",
    "Q9_comfortable_discussing_concerns": "I feel comfortable discussing my concerns with someone.",
    "Q10_family_expectations_influence": "Family expectations influence how much pressure I feel about schoolwork.",
    "Q11_sleep_on_school_nights": "How many hours do you usually sleep on school nights?",
    "Q12_screen_time_outside_schoolwork": "How much screen time do you have outside schoolwork each day?",
    "Q13_physical_activity_days": "On how many days in a typical week are you physically active?",
    "Q14_mentally_tired_or_overwhelmed": "How often do you feel mentally tired or overwhelmed?",
}
QUESTION_GROUPS = {
    "School and study experiences": list(QUESTION_OPTIONS.keys())[2:8],
    "Support and daily wellbeing": list(QUESTION_OPTIONS.keys())[8:],
}

st.set_page_config(page_title="Student Stress Assessment", page_icon="🌱", layout="centered")
st.title("🌱 Student Stress Assessment & Wellbeing Support")
st.caption("A student wellbeing capstone prototype")

if "page" not in st.session_state:
    st.session_state.page = "home"
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "sharing" not in st.session_state:
    st.session_state.sharing = {}
if "prediction" not in st.session_state:
    st.session_state.prediction = None

st.info(
    "Privacy for this demonstration: information entered here is used only during "
    "this app session. It is not saved to a database and no emails are sent. "
    "Do not enter real student or contact details in this public demonstration."
)
st.warning(
    "This is an educational prototype. Its model was trained using synthetic data "
    "and has not been validated with real students. It cannot diagnose stress or "
    "replace support from a parent, teacher, counsellor or qualified professional."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Expected: 3_ML_Model/Model_Outputs/Random_Forest.joblib")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

def go(page):
    st.session_state.page = page
    st.rerun()

def collect_answer_signals(a):
    """Readable answer-based observations; these are not model explanations."""
    signals = []
    if a["Q3_study_time_outside_school"] == "More than 3 hours":
        signals.append("You indicated that you study for more than 3 hours outside school.")
    if a["Q4_schoolwork_manageable"] in ["Disagree", "Strongly disagree"]:
        signals.append("You indicated that schoolwork does not currently feel manageable.")
    if a["Q5_think_about_tests"] in ["Often", "Very often"]:
        signals.append("You said thoughts about tests or exams come up often.")
    if a["Q6_confident_with_academic_work"] in ["Disagree", "Strongly disagree"]:
        signals.append("You indicated that you are not feeling very confident about academic work.")
    if a["Q7_compare_with_classmates"] in ["Often", "Very often"]:
        signals.append("You said you compare your academic performance with classmates often.")
    if a["Q8_difficulty_concentrating"] in ["Often", "Very often"]:
        signals.append("You reported difficulty concentrating often.")
    if a["Q9_comfortable_discussing_concerns"] in ["Disagree", "Strongly disagree"]:
        signals.append("You indicated that talking about concerns may not feel easy right now.")
    if a["Q10_family_expectations_influence"] in ["Agree", "Strongly agree"]:
        signals.append("You indicated that family expectations influence the pressure you feel.")
    if a["Q11_sleep_on_school_nights"] in ["Less than 5 hours", "5–6 hours"]:
        signals.append("You reported getting 6 hours of sleep or less on school nights.")
    if a["Q12_screen_time_outside_schoolwork"] == "More than 4 hours":
        signals.append("You reported more than 4 hours of screen time outside schoolwork.")
    if a["Q13_physical_activity_days"] == "0 days":
        signals.append("You indicated that you are not usually physically active during the week.")
    if a["Q14_mentally_tired_or_overwhelmed"] in ["Often", "Very often"]:
        signals.append("You said you feel mentally tired or overwhelmed often.")
    return signals

def suggestions(a):
    items = []
    if a["Q3_study_time_outside_school"] == "More than 3 hours":
        items.append("Try adding short breaks between study sessions. A short pause can make it easier to return with focus.")
    if a["Q4_schoolwork_manageable"] in ["Disagree", "Strongly disagree"]:
        items.append("Break a large assignment into smaller steps. If something still feels difficult, consider asking a teacher or classmate for help.")
    if a["Q5_think_about_tests"] in ["Often", "Very often"]:
        items.append("Try planning revision in smaller portions instead of thinking about the whole exam at once.")
    if a["Q6_confident_with_academic_work"] in ["Disagree", "Strongly disagree"]:
        items.append("Notice the topics you have already understood. You could also ask a teacher to help identify one area to work on next.")
    if a["Q7_compare_with_classmates"] in ["Often", "Very often"]:
        items.append("Try measuring progress against your own previous work rather than someone else's marks or pace.")
    if a["Q8_difficulty_concentrating"] in ["Often", "Very often"]:
        items.append("Try a shorter study session with notifications turned off, followed by a small break.")
    if a["Q9_comfortable_discussing_concerns"] in ["Disagree", "Strongly disagree"]:
        items.append("If talking feels difficult, you could start by writing down what is on your mind or choosing one trusted adult to approach.")
    if a["Q10_family_expectations_influence"] in ["Agree", "Strongly agree"]:
        items.append("If expectations feel heavy, consider explaining how they are affecting you to someone you trust.")
    if a["Q11_sleep_on_school_nights"] in ["Less than 5 hours", "5–6 hours"]:
        items.append("A regular wind-down routine and a consistent bedtime may help you make more room for rest.")
    if a["Q12_screen_time_outside_schoolwork"] in ["More than 4 hours", "3–4 hours"]:
        items.append("Consider taking a screen break and making some time for an offline activity you enjoy.")
    if a["Q13_physical_activity_days"] in ["0 days", "1–2 days"]:
        items.append("If it feels comfortable, add a little movement you enjoy, such as a walk, cycling or playing outdoors.")
    if a["Q14_mentally_tired_or_overwhelmed"] in ["Often", "Very often"]:
        items.append("Make space for something calming, and consider telling a trusted adult how you have been feeling.")
    return items[:3]

if st.session_state.page == "home":
    st.subheader("A little check-in about how school and everyday life are feeling")
    st.write(
        "This short questionnaire asks about schoolwork, study habits, sleep and "
        "general wellbeing. There are no right or wrong answers. Choose the answers "
        "that feel closest to your experience."
    )
    st.markdown("**What happens next?**")
    st.write("You will answer a few questions, choose whether you want to share a demonstration result, and see an experimental model estimate with general suggestions.")
    if st.button("Start check-in", type="primary", use_container_width=True):
        go("profile")

elif st.session_state.page == "profile":
    st.subheader("1. About the Student")
    st.write("The details below are for this demonstration only.")
    with st.form("profile_form"):
        full_name = st.text_input("Student's full name")
        c1, c2 = st.columns(2)
        with c1:
            student_class = st.selectbox("Class", ["Class 9", "Class 10"], index=None, placeholder="Select class")
            age = st.selectbox("Age group", QUESTION_OPTIONS["Q2_age_group"], index=None, placeholder="Select age")
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], index=None, placeholder="Select gender")
        next_profile = st.form_submit_button("Continue to consent", type="primary")
    if next_profile:
        if not full_name.strip() or not student_class or not age or not gender:
            st.warning("Please complete the name, class, age and gender fields. Gender can be set to Prefer not to say.")
        else:
            st.session_state.profile = {
                "full_name": full_name.strip(), "Q1_class": student_class,
                "Q2_age_group": age, "gender": gender
            }
            go("consent")
    if st.button("Back to welcome"):
        go("home")

elif st.session_state.page == "consent":
    st.subheader("2. Sharing Preferences and Consent")
    st.write("Choose whether you would like to share your result. You can keep it private. Your choice will not affect the questionnaire or the result.")
    st.info("This is a prototype only. No email will be sent, and the details entered here are not saved beyond this app session. Do not enter real email addresses in this public demonstration.")

    # Keep these controls outside a form so the email fields appear immediately
    # when the student changes the sharing choice.
    sharing_choice = st.radio(
        "Would you like to share your result with anyone?",
        [
            "No, keep my result private",
            "Yes, with my parent or guardian",
            "Yes, with my teacher or school counsellor",
            "Yes, with both",
        ],
        index=0,
        key="sharing_choice_widget",
    )

    parent_email = ""
    teacher_email = ""
    if sharing_choice in ["Yes, with my parent or guardian", "Yes, with both"]:
        parent_email = st.text_input(
            "Parent/guardian email address",
            placeholder="Enter parent or guardian email address",
            key="parent_email_widget",
        )
    if sharing_choice in ["Yes, with my teacher or school counsellor", "Yes, with both"]:
        teacher_email = st.text_input(
            "Teacher/counsellor email address",
            placeholder="Enter teacher or counsellor email address",
            key="teacher_email_widget",
        )

    st.markdown(
        "I understand that my questionnaire result may contain personal wellbeing "
        "information. I can choose to keep it private. This demonstration does not "
        "send or store the result."
    )
    consent_ack = st.checkbox(
        "I understand and agree to the sharing preference selected above.",
        key="consent_ack_widget",
    )

    if st.button("Continue to questionnaire", type="primary"):
        wants_share = sharing_choice != "No, keep my result private"
        if wants_share and not consent_ack:
            st.warning("Please confirm that you understand the sharing preference, or choose to keep your result private.")
        elif wants_share and (
            (sharing_choice in ["Yes, with my parent or guardian", "Yes, with both"] and not parent_email.strip())
            or (sharing_choice in ["Yes, with my teacher or school counsellor", "Yes, with both"] and not teacher_email.strip())
        ):
            st.warning("Please enter the email address for each person you selected, or choose to keep your result private.")
        else:
            st.session_state.sharing = {
                "share_parent": sharing_choice in ["Yes, with my parent or guardian", "Yes, with both"],
                "parent_email": parent_email.strip(),
                "share_teacher": sharing_choice in ["Yes, with my teacher or school counsellor", "Yes, with both"],
                "teacher_email": teacher_email.strip(),
                "consent_ack": consent_ack,
                "sharing_choice": sharing_choice,
            }
            go("questionnaire")

    if st.button("Back to student details"):
        go("profile")

elif st.session_state.page == "questionnaire":
    st.subheader("3. Wellbeing Questionnaire")
    with st.form("wellbeing_form"):
        answers = {
            "Q1_class": st.session_state.profile["Q1_class"],
            "Q2_age_group": st.session_state.profile["Q2_age_group"],
        }
        for group, keys in QUESTION_GROUPS.items():
            st.markdown(f"**{group}**")
            for key in keys:
                answers[key] = st.selectbox(
                    QUESTION_LABELS[key], QUESTION_OPTIONS[key],
                    index=None, placeholder="Select your answer", key=f"answer_{key}"
                )
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        missing = [k for k in QUESTION_OPTIONS if answers.get(k) is None]
        if missing:
            st.warning(f"Please answer all {len(missing)} remaining question(s) before continuing.")
        else:
            st.session_state.responses = answers
            try:
                # Only the original model feature columns are passed to the model.
                model_input = pd.DataFrame([{k: st.session_state.responses[k] for k in QUESTION_OPTIONS}])
                st.session_state.prediction = str(model.predict(model_input)[0]).strip().title()
                go("results")
            except Exception as exc:
                st.error(f"Prediction could not be generated: {exc}")
    if st.button("Back to sharing preferences"):
        go("consent")

elif st.session_state.page == "results":
    name = st.session_state.profile.get("full_name", "there")
    category = st.session_state.prediction
    answers = st.session_state.responses

    st.subheader("4. Your check-in result")
    st.write(f"Thank you for completing the check-in, **{name}**.")
    st.caption("This is an experimental estimate from the prototype, not a clinical assessment.")

    # A simple result card with no implied score or confidence percentage.
    with st.container(border=True):
        if category == "High":
            st.error("HIGH CATEGORY")
            meaning = "Your answers show a pattern that the model associates more closely with the High label in its synthetic training data."
            next_step = "Please consider talking to a trusted adult, such as a parent, teacher or school counsellor, about how things have been feeling."
        elif category == "Moderate":
            st.warning("MODERATE CATEGORY")
            meaning = "Your answers show a pattern that the model associates more closely with the Moderate label in its synthetic training data."
            next_step = "It may help to notice what has been feeling difficult and speak with someone you trust if you would like support."
        elif category == "Low":
            st.success("LOW CATEGORY")
            meaning = "Your answers show a pattern that the model associates more closely with the Low label in its synthetic training data."
            next_step = "Keep making time for rest, activities you enjoy and people you feel comfortable talking to. You can still ask for support whenever you need it."
        else:
            st.info(str(category))
            meaning = "The prototype returned an unexpected category."
            next_step = "Please speak with a trusted adult if you have concerns about your wellbeing."
        st.write(meaning)
        st.markdown("**A helpful next step**")
        st.write(next_step)

    st.markdown("### What your answers highlighted")
    signals = collect_answer_signals(answers)
    if signals:
        st.write("These are specific responses you selected that may be worth reflecting on. They are not a diagnosis and did not individually determine the model's result.")
        for signal in signals[:4]:
            st.markdown(f"- {signal}")
        if len(signals) > 4:
            st.caption(f"There were {len(signals) - 4} other answer areas that this prototype could flag. The list is kept short to make the result easier to read.")
    else:
        st.write("None of the answer patterns used for these reflection prompts were flagged. This does not mean that you never feel pressure or would not benefit from support.")

    st.markdown("### A few small things you could try 🌱")
    tips = suggestions(answers)
    if tips:
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.write("Keep a healthy balance between schoolwork, rest and activities you enjoy. Reach out to someone you trust whenever you need to.")

    with st.expander("How was this category estimated?"):
        st.write(
            "The app uses a Random Forest model trained on synthetic questionnaire records. "
            "During training, labels were created from a separate question about stress frequency (Q15): "
            "Never/Rarely were labelled Low, Sometimes was labelled Moderate, and Often/Very often "
            "were labelled High. The model learned patterns between answers to Q1–Q14 and those labels. "
            "It does not calculate a clinical stress score or use a validated medical threshold."
        )
        st.caption(
            "The answer highlights above are simple reflection prompts based on selected answers. "
            "They are not an explanation of the Random Forest's internal decision. Student name and "
            "gender are not used by the prediction model."
        )

    st.info(
        "Please remember: this model was trained on synthetic data and has not been validated "
        "with real students. It is for learning and demonstration only. It cannot diagnose stress. "
        "If you feel worried, overwhelmed or unsafe, speak with a trusted adult or qualified professional."
    )

    sharing = st.session_state.sharing
    if sharing.get("share_parent") or sharing.get("share_teacher"):
        if sharing.get("share_parent") and sharing.get("share_teacher"):
            choice_text = "You selected parent/guardian and teacher/counsellor."
        elif sharing.get("share_parent"):
            choice_text = "You selected parent/guardian."
        else:
            choice_text = "You selected teacher/counsellor."
    else:
        choice_text = "You chose to keep your result private."
    st.caption(f"Sharing preference: {choice_text} No email is sent by this prototype.")

    if st.button("Start a new check-in", type="primary", use_container_width=True):
        for key in ["profile", "responses", "sharing", "prediction"]:
            st.session_state[key] = {} if key != "prediction" else None
        go("home")

st.divider()
st.caption("Student Stress Assessment Capstone | Prototype only | No data is stored or emailed.")
