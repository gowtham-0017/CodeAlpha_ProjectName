import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import time
import os

# --- Custom Styling ---
st.markdown("""
    <style>
        .chat-bubble {
            max-width: 80%;
            padding: 12px 20px;
            margin: 10px 0;
            border-radius: 20px;
            font-size: 16px;
            display: inline-block;
        }
        .user-bubble {
            background-color: #d1e7dd;
            color: #000;
            align-self: flex-end;
            text-align: right;
        }
        .bot-bubble {
            background-color: #f8d7da;
            color: #000;
            align-self: flex-start;
        }
    </style>
""", unsafe_allow_html=True)

# --- Page Config ---
st.set_page_config(page_title="E-commerce Chatbot", layout="centered")
st.title("🛍️ E-commerce FAQ Chatbot")
st.markdown("<hr>", unsafe_allow_html=True)

# --- Dark/Light Mode Toggle ---
theme = st.selectbox("Select Theme:", ["🌞 Light", "🌙 Dark"])
if theme == "🌙 Dark":
    st.markdown("""
        <style>
        body, .stApp { background-color: #1e1e1e; color: white; }
        </style>
    """, unsafe_allow_html=True)

# --- FAQ Data Loader ---
def load_faq(file=None):
    if file:
        df = pd.read_csv(file)
        return dict(zip(df['question'], df['answer']))
    else:
        return {
            "What is the return policy?": "You can return products within 30 days of delivery.",
            "How do I track my order?": "Go to 'My Orders' and click on 'Track Order'.",
            "What payment methods are accepted?": "We accept credit cards, debit cards, UPI, and net banking.",
            "How can I cancel my order?": "You can cancel an order before it is shipped from the 'My Orders' page.",
            "Do you offer international shipping?": "Currently, we only ship within India."
        }

faq_file = st.file_uploader("Upload your FAQ CSV file (optional):", type="csv")
faq_data = load_faq(faq_file)

# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- Build Vector Index ---
questions = list(faq_data.keys())
answers = list(faq_data.values())
embeddings = model.encode(questions)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- Chat Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- User Input ---
query = st.text_input("Ask your question:", key="user_input")

# --- Process Query ---
if query:
    # Show user's message
    st.session_state.chat_history.append(("user", query))

    with st.spinner("🤖 Bot is typing..."):
        time.sleep(1.2)
        query_embed = model.encode([query])
        D, I = index.search(np.array(query_embed), k=1)
        matched_a = answers[I[0][0]]
        st.session_state.chat_history.append(("bot", matched_a))

        # --- Feedback Logger ---
        with st.expander("Was this helpful?"):
            feedback = st.radio("Feedback", ["👍 Yes", "👎 No"], horizontal=True, key=f"feedback_{len(st.session_state.chat_history)}")
            if feedback == "👎 No":
                comment = st.text_area("What was wrong or missing?", key=f"comment_{len(st.session_state.chat_history)}")
                if st.button("Submit Feedback", key=f"btn_{len(st.session_state.chat_history)}"):
                    with open("feedback_log.txt", "a") as f:
                        f.write(f"\nQuestion: {query}\nAnswer: {matched_a}\nFeedback: {comment}\n")
                    st.success("Thanks for your feedback!")

# --- Display Chat History ---
st.markdown("---")
st.subheader("📜 Chat History")
for sender, message in st.session_state.chat_history:
    bubble_class = "user-bubble" if sender == "user" else "bot-bubble"
    st.markdown(f"""
    <div class='chat-bubble {bubble_class}'>
        {message}
    </div>
    """, unsafe_allow_html=True)
