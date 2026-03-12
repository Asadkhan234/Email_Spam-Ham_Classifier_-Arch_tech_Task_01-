import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Custom CSS for better UI
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:bold;
    text-align:center;
    color:#4CAF50;
}
.result-spam {
    font-size:28px;
    color:red;
    font-weight:bold;
}
.result-ham {
    font-size:28px;
    color:green;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">📧 Email Spam Detection System</p>', unsafe_allow_html=True)

st.write("Detect whether an email is **Spam or Not Spam** using Machine Learning.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
"""
This app uses **Machine Learning (TF-IDF + Logistic Regression)**  
to classify emails as **Spam or Ham**.

Developer: **Asadullah Khan**
"""
)

# Email input
email_text = st.text_area(
    "✉️ Enter Email Text",
    height=200,
    placeholder="Paste your email message here..."
)

# Predict button
if st.button("🔍 Detect Spam"):

    if email_text.strip() == "":
        st.warning("⚠️ Please enter email text")

    else:
        # Transform text
        vector_input = vectorizer.transform([email_text])

        # Prediction
        prediction = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)

        # Show result
        if prediction == 1:
            st.markdown('<p class="result-spam">🚨 Spam Email Detected!</p>', unsafe_allow_html=True)
            st.progress(float(probability[0][1]))
            st.write(f"Spam Probability: **{probability[0][1]*100:.2f}%**")

        else:
            st.markdown('<p class="result-ham">✅ Not Spam (Safe Email)</p>', unsafe_allow_html=True)
            st.progress(float(probability[0][0]))
            st.write(f"Safe Probability: **{probability[0][0]*100:.2f}%**")

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit")