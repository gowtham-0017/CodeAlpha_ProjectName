# E-commerce FAQ Chatbot

A smart AI-powered chatbot built to answer frequently asked questions (FAQs) about e-commerce products.  
This project is part of the **CodeAlpha Artificial Intelligence Internship**.

---

## 📌 Project Overview
This chatbot uses **Natural Language Processing (NLP)** and **semantic search** to match user questions with the most relevant FAQs.  
It provides instant answers in a chat-like UI, enhancing customer support automation.

---

## ✅ Features
- **Smart FAQ Matching** using semantic similarity (Sentence Transformers + FAISS)  
- **Interactive Chat UI** built with **Streamlit**  
- **Real-time Search** for the best matching answers  
- **Feedback System** to improve responses  

---

## 🛠️ Technologies Used
- **Python 3.12**  
- **Streamlit** (for Chat UI)  
- **Sentence Transformers** (for semantic matching)  
- **FAISS** (for vector search)  
- **NumPy, Pandas** (for data processing)

---

## 📂 Project Structure
```
ecommerce_chatbot/
│── chatbot.py          # Main chatbot application
│── feedback_log.txt    # Stores user feedback
│── requirements.txt    # Required Python libraries
│── README.md           # Project documentation
```

---

## ⚡ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/gowtham-0017/CodeAlpha_ProjectName.git
cd CodeAlpha_ProjectName
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
streamlit run chatbot.py
```

### 4. Access in Browser
Open the link provided by Streamlit, usually:  
`http://localhost:8501`

---

## 🎯 Goal
To automate answering common e-commerce customer queries, reducing support workload and improving response time.

---

## 👤 Author
**Gowtham**  
Part of **CodeAlpha AI Internship**
