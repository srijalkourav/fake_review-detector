
import streamlit as st
import joblib
import numpy as np
import re

from scipy.sparse import hstack, csr_matrix


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FC;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #142B4A;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #142B4A;
}

div.stButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-size: 17px;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    background-color: #10558E;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL FILES
# =====================================================

@st.cache_resource
def load_models():

    model = joblib.load("fake_review_logistic_model.pkl")

    scaler = joblib.load("fake_review_scaler.pkl")

    tfidf = joblib.load("fake_review_tfidf.pkl")

    return model, scaler, tfidf


try:

    model, scaler, tfidf = load_models()

except Exception as e:

    st.error(f"Error loading model files: {e}")

    st.stop()


# =====================================================
# 4. CLEAN REVIEW TEXT
# =====================================================

def clean_review_text(text):

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# 5. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🔍 Review Analyzer")

    st.markdown("---")

    st.markdown("### About Project")

    st.write(
        "This application uses Natural Language Processing "
        "and Machine Learning to identify potentially fake "
        "product reviews."
    )

    st.markdown("---")

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - NLP
    - TF-IDF
    - Logistic Regression
    - Scikit-learn
    """)

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Algorithm:** Logistic Regression")

    st.write("**Text Processing:** TF-IDF")

    st.write("**Input:** Product Review")

    st.markdown("---")

    st.caption("AI Fake Review Detection Project")


# =====================================================
# 6. MAIN HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🔍 AI-Powered Fake Review Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze product reviews using Natural Language Processing '
    'and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# =====================================================
# 7. REVIEW INPUT
# =====================================================

st.markdown(
    '<div class="section-title">📝 Review Information</div>',
    unsafe_allow_html=True
)

review_input = st.text_area(
    "Enter Product Review",
    placeholder="Example: This product is amazing. The quality is excellent!",
    height=160
)

st.caption(
    "Enter the customer's review comment. "
    "No additional information is required."
)


# =====================================================
# 8. ANALYZE REVIEW BUTTON
# =====================================================

analyze_button = st.button(
    "🔍 Analyze Review",
    type="primary",
    use_container_width=True
)


# =====================================================
# 9. PREDICTION
# =====================================================

if analyze_button:

    if not review_input.strip():

        st.warning("Please enter a product review first.")

    else:

        cleaned_review = clean_review_text(review_input)

        if not cleaned_review:

            st.warning("Please enter valid review text.")

        else:

            try:

                # -------------------------------------
                # A. TF-IDF TRANSFORMATION
                # -------------------------------------

                review_tfidf = tfidf.transform(
                    [cleaned_review]
                )


                # -------------------------------------
                # B. AUTOMATIC METADATA CALCULATION
                # -------------------------------------

                review_length = len(cleaned_review)

                word_count = len(cleaned_review.split())

                # Default values for metadata that
                # cannot be obtained from review text.

                review_rating = 5

                number_of_helpful = 0

                number_of_photos = 0

                reviewer_activity_before = 0


                # -------------------------------------
                # C. CREATE METADATA ARRAY
                # -------------------------------------

                review_metadata = np.array([[
                    review_rating,
                    review_length,
                    word_count,
                    number_of_helpful,
                    number_of_photos,
                    reviewer_activity_before
                ]])


                # -------------------------------------
                # D. SCALE METADATA
                # -------------------------------------

                review_metadata_scaled = scaler.transform(
                    review_metadata
                )


                # -------------------------------------
                # E. CONVERT METADATA TO SPARSE MATRIX
                # -------------------------------------

                metadata_sparse = csr_matrix(
                    review_metadata_scaled
                )


                # -------------------------------------
                # F. COMBINE TF-IDF + METADATA
                # -------------------------------------

                review_combined = hstack([
                    review_tfidf,
                    metadata_sparse
                ])


                # -------------------------------------
                # G. CHECK FEATURE COUNT
                # -------------------------------------

                if review_combined.shape[1] != model.n_features_in_:

                    st.error(
                        "Feature mismatch! Check the trained model "
                        "and preprocessing files."
                    )

                    st.stop()


                # -------------------------------------
                # H. MODEL PREDICTION
                # -------------------------------------

                prediction = model.predict(
                    review_combined
                )[0]

                probabilities = model.predict_proba(
                    review_combined
                )[0]


                class_probabilities = dict(
                    zip(model.classes_, probabilities)
                )

                honest_probability = (
                    class_probabilities[0] * 100
                )

                fake_probability = (
                    class_probabilities[1] * 100
                )


                # -------------------------------------
                # I. DISPLAY PREDICTION
                # -------------------------------------

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '📌 Prediction Result'
                    '</div>',
                    unsafe_allow_html=True
                )

                if prediction == 1:

                    st.error(
                        "⚠️ This review is predicted as FAKE."
                    )

                else:

                    st.success(
                        "✅ This review is predicted as HONEST."
                    )


                # -------------------------------------
                # J. PROBABILITY RESULTS
                # -------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Honest Probability",
                        f"{honest_probability:.2f}%"
                    )

                    st.progress(
                        int(round(honest_probability))
                    )

                with col2:

                    st.metric(
                        "Fake Probability",
                        f"{fake_probability:.2f}%"
                    )

                    st.progress(
                        int(round(fake_probability))
                    )


                # -------------------------------------
                # K. REVIEW ANALYTICS
                # -------------------------------------

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '📈 Review Analytics'
                    '</div>',
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Review Length",
                        review_length
                    )

                with col2:

                    st.metric(
                        "Word Count",
                        word_count
                    )


                # -------------------------------------
                # L. ANALYZED REVIEW
                # -------------------------------------

                st.markdown("---")

                st.markdown(
                    '<div class="section-title">'
                    '📄 Analyzed Review'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.info(cleaned_review)


                # -------------------------------------
                # M. DISCLAIMER
                # -------------------------------------

                st.caption(
                    "Note: Model probabilities are estimates, "
                    "not proof that a review is fake or genuine. "
                    "Metadata values not entered by the user "
                    "are assigned default values."
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# =====================================================
# 10. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "<h3 style='text-align:center; color:#142B4A;'>"
    "AI-Powered Fake Review Detection"
    "</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:16px;'>"
    "Developed &amp; Powered by <b>Srijal Kaurav</b>"
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:13px;'>"
    "Built with Python | NLP | Machine Learning | Streamlit"
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#64748B; font-size:12px;'>"
    "© 2026 Srijal Kaurav. All rights reserved."
    "</p>",
    unsafe_allow_html=True
)
