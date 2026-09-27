import streamlit as st
import joblib

# Load your saved models and preprocessors
model = joblib.load('fake_review_logistic_model.pkl')
scaler = joblib.load('fake_review_scaler.pkl')
tfidf = joblib.load('fake_review_tfidf.pkl')

st.title("Amazon Fake Review Detection System")
st.write("Enter an Amazon product review below to check if it is predicted to be Real or Fake.")

# User input text box
review_input = st.text_area("Paste the review text here:")

if st.button("Predict"):
    if review_input.strip() == "":
        st.warning("Please enter some review text first.")
    else:
        # Transform the review text using your TF-IDF vectorizer
        data_tfidf = tfidf.transform([review_input])
        
        # If your model/scaler requires handling sparse matrices or specific features:
        # (Standard text-only pipelines use tfidf directly into the model)
        data_scaled = scaler.transform(data_tfidf)
        prediction = model.predict(data_scaled)
        
        # Display result
        if prediction[0] == 1:
            st.error("Result: This looks like a *Fake Review*!")
        else:
            st.success("Result: This looks like a *Real Review*!")