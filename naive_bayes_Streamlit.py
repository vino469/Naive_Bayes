# =========================
# naive_bayes_streamlit.py
# =========================

import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.title("🛳 Naive Bayes Classification - Whale Data")

# =========================
# Step 1: Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload analcdata_whale.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Check for target column
    if 'period_Hudson' not in df.columns:
        st.error("Target column 'period_Hudson' not found!")
    else:
        X = df.drop(columns=['period_Hudson'])
        y = df['period_Hudson']

        # Identify categorical & numeric columns
        cat_cols = [c for c in X.columns if X[c].dtype == 'object']
        num_cols = [c for c in X.columns if X[c].dtype != 'object']

        st.subheader("Target Distribution")
        st.write(y.value_counts())

        # =========================
        # Step 2: Build pipeline
        # =========================
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
                ('num', StandardScaler(), num_cols)
            ]
        )

        nb_model = GaussianNB()
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', nb_model)
        ])

        # =========================
        # Step 3: Train/test split
        # =========================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # =========================
        # Step 4: Train model
        # =========================
        model.fit(X_train, y_train)
        st.success("✅ Model trained successfully!")

        # =========================
        # Step 5: Evaluate
        # =========================
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # =========================
        # Step 6: Plot Predictions
        # =========================
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.7)
        ax.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.5)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("period_Hudson")
        ax.set_title("Naive Bayes Predictions vs Actual")
        ax.legend()
        st.pyplot(fig)

        # =========================
        # Step 7: Save model
        # =========================
        if st.checkbox("Save trained model"):
            joblib.dump(model, "naive_bayes_whale_model.pkl")
            st.success("Model saved as naive_bayes_whale_model.pkl")

        # =========================
        # Step 8: Predict new data
        # =========================
        st.subheader("Predict New Whale Data")
        st.write("Enter new data to make predictions:")

        with st.form("predict_form"):
            new_data_dict = {}
            for col in X.columns:
                if col in cat_cols:
                    new_data_dict[col] = st.text_input(f"{col} (categorical)", "Unknown")
                else:
                    new_data_dict[col] = st.number_input(f"{col} (numeric)", value=0.0)
            submit = st.form_submit_button("Predict")

        if submit:
            new_df = pd.DataFrame([new_data_dict])
            prediction = model.predict(new_df)
            st.success(f"Predicted Class: {prediction[0]}")
