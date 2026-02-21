import streamlit as st
import joblib
import pandas as pd
import re
import PyPDF2

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="HR Multi-Bias Detection System",
    page_icon="🚀",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>
body { background-color: #0E1117; }
h1 { color: #00BFFF; text-align: center; }
h2, h3 { color: white; }
.stTextArea textarea {
    background-color: #1E2228 !important;
    color: white !important;
    border-radius: 10px;
}
.stButton > button {
    background: linear-gradient(90deg, #00BFFF, #0072FF);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}
.stFileUploader {
    background-color: #1E2228;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- SUGGESTION DICTIONARY ---------------- #

suggestions = {
    "dominant": "proactive",
    "aggressive": "driven",
    "competitive": "goal-oriented",
    "assertive": "clear communicator",
    "confident": "self-assured",
    "ambitious": "motivated",
    "strong": "capable",
    "nurturing": "supportive",
    "empathetic": "understanding",
    "compassionate": "considerate",
    "collaborative": "team-oriented",
    "young": "skilled",
    "youthful": "qualified",
    "energetic": "motivated",
    "graduate": "candidate",
    "overqualified": "highly experienced",
    "mandatory": "preferred",
    "minimum": "at least"
}

# ---------------- HELPER FUNCTIONS ---------------- #

def clean_word(word):
    return re.sub(r'[^a-zA-Z]', '', word.lower()).rstrip("s")

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def analyze_text(text):

    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]

    confidence_dict = dict(zip(model.classes_, probabilities))

    # ----------- DASHBOARD METRICS ----------- #

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Bias", prediction.replace("_", " ").title())
    with col2:
        st.metric("Confidence", f"{max(probabilities)*100:.2f}%")

    st.markdown("---")

    # ----------- MULTI BIAS SCORES ----------- #

    st.subheader("📊 Multi-Bias Probability Scores")

    bias_df = pd.DataFrame({
        "Bias Type": [k.replace('_', ' ').title() for k in confidence_dict.keys()],
        "Score (%)": [v * 100 for v in confidence_dict.values()]
    })

    st.dataframe(bias_df)
    st.bar_chart(bias_df.set_index("Bias Type"))

    st.markdown("---")

    # ----------- HIGHLIGHTED TEXT ----------- #

    st.subheader("🖍 Highlighted Biased Words")
    highlighted = []
    for word in text.split():
        if clean_word(word) in suggestions:
            highlighted.append(f":red[{word}]")
        else:
            highlighted.append(word)
    st.markdown(" ".join(highlighted))

    st.markdown("---")

    # ----------- NEUTRAL REWRITE ----------- #

    st.subheader("✏ Rewritten Neutral Version")
    rewritten = []
    for word in text.split():
        cw = clean_word(word)
        if cw in suggestions:
            rewritten.append(suggestions[cw])
        else:
            rewritten.append(word)
    st.write(" ".join(rewritten))

    st.markdown("---")

    # ----------- TOP TF-IDF WORDS ----------- #

    st.subheader("🧠 Top Influential Words (TF-IDF)")
    feature_names = vectorizer.get_feature_names_out()
    tfidf_array = text_tfidf.toarray()[0]
    top_indices = tfidf_array.argsort()[-5:][::-1]

    for index in top_indices:
        st.write(f"{feature_names[index]} (score: {tfidf_array[index]:.3f})")


# ---------------- SIDEBAR NAVIGATION ---------------- #

st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Select Input Type",
    ("Text Analysis", "PDF Analysis", "CSV Bulk Analysis")
)

st.markdown("# 🚀 HR Multi-Bias Detection System")
st.markdown("### Detect Gender, Age & Experience Bias using Machine Learning")
st.markdown("---")

# ---------------- TEXT ANALYSIS ---------------- #

if option == "Text Analysis":
    st.subheader("🔍 Single Job Description Analysis")
    text = st.text_area("Paste Job Description Here")
    if st.button("Analyze Text"):
        if text.strip():
            analyze_text(text)
        else:
            st.warning("Please enter text.")

# ---------------- PDF ANALYSIS ---------------- #

elif option == "PDF Analysis":
    st.subheader("📄 Upload PDF for Analysis")
    uploaded_pdf = st.file_uploader("Upload Resume / Job Description PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        if pdf_text.strip():
            st.subheader("📜 Extracted Text Preview")
            st.write(pdf_text[:1000])
            analyze_text(pdf_text)
        else:
            st.warning("Could not extract text from PDF.")

# ---------------- CSV BULK ANALYSIS ---------------- #

elif option == "CSV Bulk Analysis":
    st.subheader("📂 Bulk CSV Analysis")
    uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            tfidf_data = vectorizer.transform(df["text"])
            predictions = model.predict(tfidf_data)
            df["Predicted Bias"] = predictions
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="bias_results.csv",
                mime="text/csv",
            )
        else:
            st.error("CSV must contain a column named 'text'")