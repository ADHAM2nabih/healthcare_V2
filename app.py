# app.py
import streamlit as st
import requests
import random
import re
import joblib
import pickle
import os

# ====== Load Models and Encoders ======
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder(path):
    return joblib.load(path)

model_paths = {
    '1': 'machine learning/mlp_pipeline_model_disease_data.pkl',
    '2a': 'machine learning/logistic_pipeline_model_disease_data.pkl',
    '2b': 'machine learning/mlp_pipeline_model_disease_data_train-00000-of-0000.pkl',
    '3': 'machine learning/mlp_pipeline_model.pkl'
}

encoder_paths = {
    '1': 'label_encoder/label_encoder_disease_data.joblib',
    '2': 'label_encoder/label_encoder_train-00001-of-00004.joblib',
    '3': 'label_encoder/label_encoder_machine.joblib'
}

models = {
    '1': load_model(model_paths['1']),
    '2a': load_model(model_paths['2a']),
    '2b': load_model(model_paths['2b']),
    '3': load_model(model_paths['3'])
}

encoders = {
    '1': load_encoder(encoder_paths['1']),
    '2': load_encoder(encoder_paths['2']),
    '3': load_encoder(encoder_paths['3'])
}

# ====== API Request Function ======
def get_next_question(user_input):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-website.com",
        "X-Title": "Arabic Chatbot"
    }
    payload = {
        "model": "qwen/qwen-2.5-7b-instruct:free",
        "messages": [{"role": "user", "content": user_input}]
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        bot_reply = data['choices'][0]['message']['content']
        return re.sub(r'[{}]', '', re.sub(r'\\boxed\\s*', '', bot_reply)).strip()
    except requests.exceptions.RequestException as e:
        return f"⚠️ Connection error: {str(e)}"
    except KeyError:
        return "⚠️ Error processing server response"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# ====== Streamlit UI ======
st.title("🧠 AI Healthcare Assistant")

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.qa_pairs = []
    st.session_state.max_qs = 0

if st.session_state.step == 0:
    st.session_state.max_qs = st.number_input("How many questions would you like to answer?", min_value=1, max_value=10, step=1)
    symptoms = st.text_input("What symptoms are you experiencing? (Example: headache, dizziness)")
    if st.button("Start Diagnosis") and symptoms:
        st.session_state.qa_pairs.append(symptoms)
        st.session_state.step += 1

elif st.session_state.step <= st.session_state.max_qs:
    prompt = "\n".join(st.session_state.qa_pairs)
    question = get_next_question(prompt)
    st.subheader(f"Question {st.session_state.step}:")
    answer = st.text_input("", key=f"answer_{st.session_state.step}")
    if st.button("Next") and answer:
        st.session_state.qa_pairs.append(answer)
        st.session_state.step += 1

else:
    st.success("✅ The questions are complete. Analyzing your health status now...")

    input_data = [" ".join(st.session_state.qa_pairs)]

    chosen_model_key = random.choice(list(models.keys()))
    model = models[chosen_model_key]

    if chosen_model_key.startswith('1'):
        encoder = encoders['1']
    elif chosen_model_key.startswith('2'):
        encoder = encoders['2']
    else:
        encoder = encoders['3']

    try:
        prediction = model.predict(input_data)
        disease = encoder.inverse_transform(prediction)[0]
        st.subheader(f"🔍 Predicted Disease: {disease}")
        st.info("💡 Temporary advice: Please rest and drink plenty of fluids until you visit a doctor.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

    if st.button("Restart"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
