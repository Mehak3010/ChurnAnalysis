import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df

df = load_data()

numerical_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

@st.cache_resource
def train_prediction_model(df):

    df_full = df.drop('customerID', axis=1)

    X_full = df_full.drop('Churn', axis=1)
    y_full = df_full['Churn']

    le = LabelEncoder()
    y_full = le.fit_transform(y_full)

    categorical_cols = X_full.select_dtypes(
        include=['object']
    ).columns

    X_encoded = pd.get_dummies(
        X_full,
        columns=categorical_cols,
        drop_first=True
    )

    scaler = StandardScaler()

    numerical_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]


    X_encoded[numerical_cols] = scaler.fit_transform(
        X_encoded[numerical_cols]
    )

    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_encoded, y_full)

    return model, scaler, X_encoded.columns

prediction_model, prediction_scaler, model_columns = train_prediction_model(df)


# tab selection
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive  Summary",
    "Customer Insights",
    "Model Performance",
    "Predict Customer"
])
# Main title
st.markdown("""
<div style='padding:35px;box-shadow:0 8px 20px rgba(0,0,0,0.2);margin-bottom:30px;border-radius:15px;
background:linear-gradient(90deg,#1E3A8A,#2563EB);color:white;text-align:center;'>

<h1>📈 Customer Retention Intelligence Platform</h1>

<p>Machine Learning Powered Customer Churn Prediction & Business Insights</p>

</div>
""", unsafe_allow_html=True)

with tab1:

    st.header("📊 Executive Summary")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Customer Base",
            f"{len(df):,}"
        )

    with col2:
        churn_rate = (
            (df['Churn'] == 'Yes').mean() * 100
        )

        st.metric(
            "⚠️ Customers at Risk",
            f"{churn_rate:.2f}%"
        )

    with col3:
        avg_tenure = df['tenure'].mean()

        st.metric(
            "📅 Avg. Tenure",
            f"{avg_tenure:.1f} months"
        )

    with col4:
        avg_monthly = df['MonthlyCharges'].mean()

        st.metric(
            "💰 Avg. Monthly Revenue",
            f"${avg_monthly:.2f}"
        )

    # Two-column layout
    col_left, col_right = st.columns([1, 1])

    with col_left:

        st.info("""
    ```

    ### 🔍 Executive Insights

    • Approximately **26% of customers** are at risk of churn.

    • Customers with **month-to-month contracts** exhibit significantly higher churn rates.

    • Customers with **short tenure periods** are more likely to leave.

    • Early intervention strategies can substantially improve customer retention.
    """)

    with col_right:

        st.subheader("📋 Dataset Snapshot")

        st.dataframe(
            df.head(5),
            use_container_width=True,
            height=250
        )

    # Quick Business Overview
    st.success("""
    ```

    ### 📌 Business Objective

    Develop a machine learning-powered customer retention intelligence platform to:

    * Identify customers likely to churn.
    * Understand key drivers influencing churn.
    * Compare predictive models.
    * Support proactive retention strategies through actionable recommendations.
    """)


with tab2:

    st.header("📊 Customer Insights")

    # Key Findings
    st.success("""
    ```

    ### 🔍 Key Findings

    ✔ Month-to-month contracts drive the highest churn.

    ✔ Customers with tenure below 12 months are most vulnerable.

    ✔ Higher monthly charges correlate with increased churn risk.

    ✔ Long-term contracts contribute positively to customer retention.
    """)

    # First Row
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:

        st.subheader("👥 Who Is Leaving?")

        fig_churn = px.pie(
            df,
            names='Churn',
            title='Customer Churn Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_churn.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=320
        )

        st.plotly_chart(
            fig_churn,
            use_container_width=True
        )

    with row1_col2:

        st.subheader("📑 Which Contracts Retain Customers?")

        fig_contract = px.histogram(
            df,
            x='Contract',
            color='Churn',
            barmode='group',
            title='Churn by Contract Type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_contract.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=320
        )

        st.plotly_chart(
            fig_contract,
            use_container_width=True
        )

    # Second Row
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:

        st.subheader("⏳ When Do Customers Leave?")

        fig_tenure = px.histogram(
            df,
            x='tenure',
            color='Churn',
            barmode='overlay',
            nbins=30,
            title='Tenure Distribution by Churn',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_tenure.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=320
        )

        st.plotly_chart(
            fig_tenure,
            use_container_width=True
        )

    with row2_col2:

        st.subheader("💰 Do Higher Charges Increase Churn?")

        fig_monthly = px.histogram(
            df,
            x='MonthlyCharges',
            color='Churn',
            barmode='overlay',
            nbins=30,
            title='Monthly Charges Distribution by Churn',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig_monthly.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            height=320
        )

        st.plotly_chart(
            fig_monthly,
            use_container_width=True
        )

    
with tab3:

    st.header("🤖 Model Performance")

    # -------------------------
    # Feature Engineering
    # -------------------------
    df_model = df.drop('customerID', axis=1)

    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']

    le = LabelEncoder()
    y = le.fit_transform(y)

    categorical_cols = X.select_dtypes(
        include=['object']
    ).columns

    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train[numerical_cols] = scaler.fit_transform(
        X_train[numerical_cols]
    )

    X_test[numerical_cols] = scaler.transform(
        X_test[numerical_cols]
    )

    # -------------------------
    # Model Selection
    # -------------------------
    st.subheader("⚙️ Model Selection")

    model_option = st.radio(
        "Choose a model:",
        [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        horizontal=True
    )

    if model_option == "Logistic Regression":

        model = LogisticRegression(
            random_state=42
        )

    elif model_option == "Random Forest":

        model = RandomForestClassifier(
            random_state=42
        )

    else:

        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # Selected Model
    # -------------------------
    st.success(
        f"🏆 Selected Model: {model_option}"
    )

    # -------------------------
    # Metrics
    # -------------------------
    st.subheader("📈 Performance Metrics")

    row1 = st.columns(3)

    with row1[0]:
        st.metric(
            "Accuracy",
            f"{accuracy_score(y_test, y_pred):.4f}"
        )

    with row1[1]:
        st.metric(
            "Precision",
            f"{precision_score(y_test, y_pred):.4f}"
        )

    with row1[2]:
        st.metric(
            "Recall",
            f"{recall_score(y_test, y_pred):.4f}"
        )

    row2 = st.columns(2)

    with row2[0]:
        st.metric(
            "F1 Score",
            f"{f1_score(y_test, y_pred):.4f}"
        )

    with row2[1]:
        st.metric(
            "ROC-AUC",
            f"{roc_auc_score(y_test, y_pred_proba):.4f}"
        )

    # -------------------------
    # Feature Importance
    # -------------------------
    st.subheader("🔍 Feature Importance")

    if model_option != "Logistic Regression":

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig_importance = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Top Factors Influencing Customer Churn"
        )

        fig_importance.update_layout(
            height=300,
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance is available for Random Forest and XGBoost models."
        )

    # -------------------------
    # Detailed Reports
    # -------------------------
    with st.expander("🎯 View Confusion Matrix"):

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        fig_cm = px.imshow(
            cm,
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            },
            x=['No Churn', 'Churn'],
            y=['No Churn', 'Churn'],
            color_continuous_scale='Blues',
            text_auto=True
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

    with st.expander("📋 View Classification Report"):

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True,
            height=250
        )

        st.success("""
    ```

    ### Business Interpretation

    A higher recall score ensures that customers likely to churn are identified early, enabling proactive retention strategies.
    """)
                
with tab4:

    st.header("🔮 Customer Churn Predictor")

    st.markdown(
        "Predict customer churn risk and receive actionable retention recommendations."
    )

    # -------------------------
    # Customer Profile Inputs
    # -------------------------
    st.subheader("📝 Customer Profile")

    with st.expander("👤 Personal Information", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

        with col2:
            partner = st.selectbox(
                "Partner",
                ["No", "Yes"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"]
            )

    with st.expander("📞 Service Details", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox(
                "Phone Service",
                ["No", "Yes"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

        with col2:
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

    with st.expander("💳 Subscription Details", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            tenure = st.slider(
                "Tenure (months)",
                0,
                72,
                12
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

        with col2:
            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["No", "Yes"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=50.0
            )

    total_charges = tenure * monthly_charges

    # -------------------------
    # Input Data
    # -------------------------
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    left_col, right_col = st.columns([2, 1])

    with right_col:

        st.subheader("🔍 Customer Summary")

        st.dataframe(
            input_data,
            use_container_width=True,
            height=250,
            hide_index=True
        )


    predict = st.button(
    "🚀 Predict Churn Risk",
    use_container_width=True,
    type="primary"
    )


    if predict:

        df_full = df.drop('customerID', axis=1)

        X_full = df_full.drop('Churn', axis=1)

        combined = pd.concat(
            [X_full, input_data],
            axis=0
        )

        categorical_cols = combined.select_dtypes(
            include=['object']
        ).columns

        combined = pd.get_dummies(
            combined,
            columns=categorical_cols,
            drop_first=True
        )

        input_encoded = combined.iloc[-1:]

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )
        
        input_encoded[numerical_cols] = prediction_scaler.transform(
            input_encoded[numerical_cols]
        )

        prediction = prediction_model.predict(input_encoded)

        prediction_proba = prediction_model.predict_proba(input_encoded)

        churn_prob = prediction_proba[0][1] * 100

        st.markdown("---")
        
        st.success("Prediction completed successfully.")
        st.subheader("📊 Prediction Results")

        gauge_col, result_col = st.columns([2, 1])

        with gauge_col:

            st.subheader("📈 Churn Risk Gauge")

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with result_col:

            st.subheader("📊 Prediction Result")

            st.metric(
                "Churn Probability",
                f"{churn_prob:.2f}%"
            )

            if prediction[0] == 1:

                st.error(
                    "⚠️ Likely to Churn"
                )

            else:

                st.success(
                    "✅ Likely to Stay"
                )

        # -------------------------
        # Recommendations
        # -------------------------
        st.subheader("💡 Recommended Actions")

        if churn_prob >= 70:

            st.markdown("""
            ### Immediate Actions

            • Offer personalized retention discounts.

            • Assign a customer success representative.

            • Recommend annual subscription upgrades.

            • Conduct a customer satisfaction follow-up.

            • Prioritize intervention to prevent revenue loss.
            """)

        elif churn_prob >= 40:

           st.markdown("""
            ### Preventive Actions

            • Offer loyalty rewards.

            • Recommend bundled services.

            • Increase engagement through targeted campaigns.

            • Monitor future customer interactions.
            """)

        else:

            st.markdown("""
            ### Maintenance Actions

            • Continue delivering excellent service.

            • Encourage referrals and testimonials.

            • Promote premium offerings.

            • Maintain regular customer engagement.
            """)
