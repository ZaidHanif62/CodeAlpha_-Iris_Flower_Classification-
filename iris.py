import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Iris Flower Classification", layout="wide")
st.markdown("<h1 style='text-align: center; color: #8e44ad;'>Iris Flower Classification App </h1>", unsafe_allow_html=True)
st.write("### Train ML Models to classify Iris flower species using their measurements.")
uploaded_file = st.file_uploader("Upload your Iris dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show dataset
    with st.expander("View Dataset"):
        st.dataframe(df)

    # Features & Target
    X = df.iloc[:, :-1]   
    y = df.iloc[:, -1]    

    if y.dtypes == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    test_size = st.sidebar.slider("Test Data Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )

    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox("Choose a Classifier", ["Decision Tree", "Logistic Regression"])

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Results
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"## Model: {model_choice}")
    st.markdown(f"### Accuracy: **{acc:.2f}**")

    # Classification Report
    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap="Purples"))

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="viridis", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_choice}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to proceed.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed with using Streamlit</p>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #8e44ad;'>👨‍💻 Zaid Hanif</h4>", 
    unsafe_allow_html=True
)
