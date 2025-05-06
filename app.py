import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import random
from nltk.corpus import stopwords
from collections import Counter
import time
import requests  # new

# ——— Hugging Face model repo identifier ———
MODEL_REPO = "veyi/mental-health-chatbot"

@st.cache_resource
def load_model_and_tokenizer():
    # load from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    # download label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

# ——— Load Model and Tokenizer ———
try:
    model, tokenizer, label_encoder = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop()

# ——— Load Stopwords ———
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    st.warning("NLTK stopwords not found. Downloading…")
    import nltk
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ——— Cleaning Function ———
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r"[^\w\s]", "", statement)
    statement = re.sub(r"\d+", "", statement)
    words = statement.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# … rest of your responses, follow‑up dicts, and Streamlit UI code unchanged …


# --- Responses Dictionary ---
responses = {
    "Depression": [
        "I'm really sorry you're feeling this way. You're not alone, and it's okay to ask for help. 💜",
        "It’s brave of you to share your feelings. Remember, even the darkest nights end with sunrise. 🌅",
        "You deserve kindness, especially from yourself. Would you like to talk about what’s been hurting most? 💙"
    ],
    "Borderline personality disorder": [
        "It sounds like you're experiencing intense emotions. You matter, and you deserve relationships that feel safe and supportive. 💙",
        "Your feelings are valid. Even when emotions feel overwhelming, you’re not alone. 🫂",
        "You are worthy of stability and peace. Let's talk more if you want. 🌿"
    ],
    "Bipolar": [
        "It seems like your moods may be swinging. You're not alone, and finding stability is possible with the right support. 💚",
        "Managing highs and lows takes incredible strength. I'm here with you. 💫",
        "Even in chaos, there can be calm. Would you like to share how you’ve been feeling lately? 🌻"
    ],
    "Anxiety": [
        "Feeling overwhelmed can be really difficult. You're doing your best, and that’s enough. 💙",
        "You are stronger than your worries. Let's take it one small step at a time. 🌼",
        "Anxiety may whisper fears, but you hold the power to overcome them. Shall we talk more? 🌿"
    ],
    "Mentalillness": [
        "Living with mental health challenges isn’t easy, but you’re stronger than you think. 💛",
        "Every small effort you make matters. Your journey deserves respect and care. 🛤️",
        "I'm proud of you for showing up today. Would you like to share more about your experience? 🌸"
    ],
    "Schizophrenia": [
        "It’s okay if you’re feeling confused or out of touch sometimes. You’re not broken—you deserve understanding and compassionate care. 💜",
        "Your experiences are valid, even when others may not understand them. I’m here for you. 🌈",
        "You're not alone. Would you like to talk about what’s been most confusing or challenging? 🧩"
    ],
    "Normal": [
        "You seem happy and balanced! It's wonderful to see you taking care of yourself. 🌟",
        "It’s great that you're feeling good! Remember to keep nurturing your well-being. 🌸",
        "You’re doing great! Even small positive steps matter a lot. Keep shining. ☀️"
    ],
    "Personality disorder": [
        "You may feel misunderstood at times, but your feelings are valid. You're not alone—there’s help available. 💙",
        "Building healthy relationships can be challenging but possible. You are worthy of connection. 🤝",
        "Your experiences matter. Would you like to talk more about what’s been difficult recently? 🌿"
    ],
    "Suicidal": [
        "I'm really concerned about you. You matter deeply. Please reach out to someone you trust or a mental health professional. ❤️",
        "Your life is precious, even when it feels heavy. You are not alone in this. 🕊️",
        "I hear you, and your pain is real. Would you like to talk about anything that could bring a little comfort now? 🧡"
    ],
    "Stress": [
        "Stress can feel like a heavy weight. Have you had a chance to rest or take care of yourself today? 💚",
        "It’s okay to slow down. You deserve moments of peace and breathing space. 🌿",
        "One step at a time—you are doing the best you can. Would you like to talk about ways to ease the pressure? ☁️"
    ]
}

# --- Follow-up Questions Dictionary ---
follow_up_questions_by_mood = {
    "Normal": [
        "What has been something positive for you lately?",
        "When do you feel the most at peace?",
        "Is there a hobby or activity you've been enjoying?",
        "What's something you're grateful for today?"
    ],
    "Anxiety": [
        "Would you like to talk about what's worrying you the most?",
        "What usually helps you feel calmer during stressful times?",
        "Is there a safe space where you feel relaxed?",
        "Have you tried any breathing exercises or grounding techniques?"
    ],
    "Depression": [
        "Would you like to share what's been weighing heavily on you?",
        "Have you been able to find anything that brings you comfort?",
        "What’s been the hardest part for you recently?",
        "Is there someone you feel safe opening up to?"
    ],
    "Suicidal": [
        "I'm really concerned about you. Would you like to share what's been hurting the most?",
        "You're important. What’s one thing you wish others understood about your pain?",
        "Who in your life might offer you support right now?",
        "What would help you feel a little bit safer today?"
    ],
    "Bipolar": [
        "How have your energy levels been lately?",
        "When you're feeling very up or down, what helps you the most?",
        "Would you like to talk about recent mood changes you've noticed?",
        "How do you usually cope during emotional highs or lows?"
    ],
    "Schizophrenia": [
        "Have you been experiencing anything confusing or distressing lately?",
        "Would you like to talk about what’s been most challenging for you?",
        "Is there anything that helps you feel more connected to reality?",
        "I'm here for you. Would you like to share how you’ve been feeling?"
    ],
    "Borderline personality disorder": [
        "What emotions have been feeling strongest for you recently?",
        "Would you like to talk about any recent relationship experiences?",
        "Is there something that helps you feel more grounded during emotional times?",
        "When do you feel most understood and supported?"
    ],
    "Personality disorder": [
        "Have your relationships felt difficult lately?",
        "Would you like to share about times you felt misunderstood?",
        "Is there something that helps you feel safe and stable?",
        "How have your emotions been affecting you recently?"
    ],
    "Mentalillness": [
        "Would you like to talk about your daily challenges?",
        "Have you found any coping strategies that work for you?",
        "Is there something that gives you hope even during tough times?",
        "You're doing your best — would you like to share how you manage your days?"
    ],
    "Stress": [
        "What's been the biggest source of stress for you lately?",
        "Have you had a chance to take a break recently?",
        "Would you like to share how you usually cope with stress?",
        "Is there something small you can do today to take care of yourself?"
    ]
}

# --- Streamlit UI ---
st.title("🧠 Mental Health Chatbot")
st.write("Tell me how you’re feeling. I'm here to listen. 💬")
st.write("Please remember, this is not a substitute for professional medical advice.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("How are you feeling right now?", key="user_input")

col1, col2 = st.columns([3, 1])  # Create two columns for layout

with col1:
    if st.button("Analyze"):
        if user_input.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            st.markdown(f"**You:** {user_input}")

            happy_words = ["happy", "good", "great", "awesome", "amazing", "joyful", "excited", "well", "fine"]
            if any(word in user_input.lower() for word in happy_words):
                predicted_label = "Normal"
            else:
                try:
                    cleaned = clean_statement(user_input)
                    inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        pred_class = torch.argmax(logits, dim=1).item()
                        predicted_label = label_encoder.inverse_transform([pred_class])[0]
                except Exception as e:
                    st.error(f"Something went wrong during analysis: {str(e)}")
                    predicted_label = None

            if predicted_label:
                st.write(f"**Predicted Condition:** `{predicted_label}`")
                reply = random.choice(responses.get(predicted_label, ["I'm here to listen. 💙"]))
                st.success(f"🤖 **Chatbot:** {reply}")
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

            # Optional: Display a follow-up question
                follow_up = random.choice(follow_up_questions_by_mood.get(predicted_label, ["Is there anything else you'd like to share?"]))
                st.info(f"🤖 **Chatbot (Follow-up):** {follow_up}")
                st.session_state["chat_history"].append({"role": "assistant", "content": follow_up})

        #st.session_state["user_input"] = "" # Clear the input field

with col2:
    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []

# --- Display Chat History ---
st.subheader("Chat History:")
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Important Disclaimer ---
st.markdown("---")
st.warning("This chatbot provides general information and is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis, please seek help from a qualified healthcare professional or a crisis hotline immediately.")