
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Page Config
st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS
st.markdown("""
    <style>
        .stApp { background-color: #f8fafc; }
        header { visibility: hidden; }
        .block-container { padding: 0.5rem 4rem !important; max-width: 1600px !important; }
        
        .card {
            background: white;
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        
        .header-bar {
            background: white;
            padding: 18px 48px;
            margin: -0.5rem -4rem 1.5rem -4rem;
            display: flex;
            align-items: center;
            gap: 18px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .logo {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 22px;
        }
        
        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }
        
        .metric-card.blue { border-left: 4px solid #3b82f6; background: linear-gradient(90deg, #f0f9ff 0%, white 100%); }
        .metric-card.red { border-left: 4px solid #ef4444; background: linear-gradient(90deg, #fef2f2 0%, white 100%); }
        .metric-card.yellow { border-left: 4px solid #f59e0b; background: linear-gradient(90deg, #fffbeb 0%, white 100%); }
        .metric-card.green { border-left: 4px solid #10b981; background: linear-gradient(90deg, #f0fdf4 0%, white 100%); }
        
        .stTabs [data-baseweb="tab-list"] { 
            gap: 8px;
            background-color: #f1f5f9;
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 18px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            height: 40px;
            padding: 0 22px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] { background-color: white; color: #1e293b; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        
        .stRadio [data-baseweb="radio-group"] { gap: 12px; }
        .stRadio [data-baseweb="radio"] { padding:6px 10px; border:1px solid #e2e8f0; border-radius:10px; background-color:white; }
        .stRadio [aria-checked="true"] [data-baseweb="radio-mark"] { background-color:#6366f1 !important; border-color:#6366f1 !important; }
        .stRadio div[data-testid="stMarkdownContainer"] p { margin:0 !important; }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    return df

df = load_data()

# Train models and get metrics
@st.cache_data
def get_model_metrics():
    df_model = df.drop(['customerID'], axis=1, errors='ignore')
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
    return metrics

model_metrics = get_model_metrics()

# Metrics for overview
total = len(df)
churned = (df['Churn'] == 'Yes').sum()
churn_rate = (churned / total) * 100
revenue_lost = df[df['Churn'] == 'Yes']['TotalCharges'].sum()
avg_tenure = df['tenure'].mean()

# Header
st.markdown("""
    <div class="header-bar">
        <div class="logo">📈</div>
        <div>
            <h1 style="font-size: 22px; margin: 0; color: #1e293b; font-weight: 700;">Churn Analytics</h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 14px;">Telco Customer Insights</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Top Metrics FIRST
m1, m2, m3, m4 = st.columns(4, gap="medium")
with m1:
    st.markdown(f"""
        <div class="metric-card blue">
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Total Customers</p>
                <p style="margin:10px 0 0 0;font-size:30px;font-weight:800;color:#1e293b;">{total:,}</p>
            </div>
            <div style="width:44px;height:44px;background-color:#dbeafe;border-radius:14px;display:flex;align-items:center;justify-content:center;">
                <span style="font-size:22px;">👥</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
        <div class="metric-card red">
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Churn Rate</p>
                <p style="margin:10px 0 0 0;font-size:30px;font-weight:800;color:#ef4444;">{churn_rate:.1f}%</p>
            </div>
            <div style="width:44px;height:44px;background-color:#fee2e2;border-radius:14px;display:flex;align-items:center;justify-content:center;">
                <span style="font-size:22px;">🚶</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
        <div class="metric-card yellow">
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Revenue Lost</p>
                <p style="margin:10px 0 0 0;font-size:30px;font-weight:800;color:#f59e0b;">${revenue_lost/1000000:.1f}M</p>
            </div>
            <div style="width:44px;height:44px;background-color:#fef3c7;border-radius:14px;display:flex;align-items:center;justify-content:center;">
                <span style="font-size:22px;">💸</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
        <div class="metric-card green">
            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Avg Tenure</p>
                <p style="margin:10px 0 0 0;font-size:30px;font-weight:800;color:#10b981;">{avg_tenure:.0f} mo</p>
            </div>
            <div style="width:44px;height:44px;background-color:#dcfce7;border-radius:14px;display:flex;align-items:center;justify-content:center;">
                <span style="font-size:22px;">⏱️</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="height: 24px;"></div>
""", unsafe_allow_html=True)

# TABS NOW (below metrics, above graphs)
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Churn Drivers", "Model", "Insights"])

with tab1:
    # Row 1: Churn & Contract Graphs
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        st.markdown("""
            <div class="card">
                <h3 style="font-size: 16px; margin: 0 0 4px 0; color: #1e293b; font-weight: 700;">Customer Status</h3>
                <p style="margin:0 0 20px 0;color:#64748b;font-size:13px;">Churn vs Retention</p>
        """, unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[total - churned, churned],
            hole=0.7,
            marker=dict(colors=['#06b6d4', '#ef4444']),
            textinfo='none'
        )])
        fig.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=12)), margin=dict(t=0, b=60, l=0, r=0), height=220)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
            <div class="card">
                <h3 style="font-size: 16px; margin: 0 0 4px 0; color: #1e293b; font-weight: 700;">Contract Type</h3>
                <p style="margin:0 0 20px 0;color:#64748b;font-size:13px;">Month-to-month has highest churn</p>
        """, unsafe_allow_html=True)
        contract = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
        contract_total = df.groupby('Contract').size().reset_index(name='total')
        contract = contract.merge(contract_total, on='Contract')
        contract['pct'] = (contract['count'] / contract['total']) * 100
        contract = contract[contract['Churn'] == 'Yes']
        fig = px.bar(contract, x='Contract', y='pct', color='Contract', color_discrete_map={'Month-to-month':'#ef4444','One year':'#f59e0b','Two year':'#06b6d4'})
        fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='', yaxis=dict(range=[0,60]), margin=dict(t=0,b=0,l=0,r=0), height=220, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
        <div class="card">
            <h3 style="font-size: 16px; margin: 0 0 4px 0; color: #1e293b; font-weight: 700;">Tenure vs Churn</h3>
            <p style="margin:0 0 20px 0;color:#64748b;font-size:13px;">New customers are most likely to churn</p>
    """, unsafe_allow_html=True)
    df['tenure_bin'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72], labels=['0-12','13-24','25-36','37-48','49-60','61-72'])
    tenure_churn = df.groupby('tenure_bin')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    fig = px.line(tenure_churn, x='tenure_bin', y='Churn', markers=True, color_discrete_sequence=['#7c3aed'])
    fig.update_layout(yaxis_title='Churn (%)', xaxis_title='Tenure (months)', yaxis=dict(range=[0,50]), margin=dict(t=0,b=0,l=0,r=0), height=260, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
            <div class="card">
                <h3 style="font-size: 16px; margin: 0 0 4px 0; color: #1e293b; font-weight: 700;">Internet Service</h3>
                <p style="margin:0 0 20px 0;color:#64748b;font-size:13px;">Fiber optic churns most</p>
        """, unsafe_allow_html=True)
        internet = df.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        internet_total = df.groupby('InternetService').size().reset_index(name='total')
        internet = internet.merge(internet_total, on='InternetService')
        internet['pct'] = (internet['count'] / internet['total']) * 100
        internet = internet[internet['Churn'] == 'Yes']
        fig = px.bar(internet, y='InternetService', x='pct', color='InternetService', color_discrete_sequence=['#3b82f6', '#2563eb', '#1d4ed8'], orientation='h')
        fig.update_layout(showlegend=False, xaxis_title='Churn (%)', yaxis_title='', xaxis=dict(range=[0,60]), margin=dict(t=0,b=0,l=0,r=0), height=210, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
            <div class="card">
                <h3 style="font-size: 16px; margin: 0 0 4px 0; color: #1e293b; font-weight: 700;">Payment Method</h3>
                <p style="margin:0 0 20px 0;color:#64748b;font-size:13px;">Electronic check is riskiest</p>
        """, unsafe_allow_html=True)
        payment = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='count')
        payment_total = df.groupby('PaymentMethod').size().reset_index(name='total')
        payment = payment.merge(payment_total, on='PaymentMethod')
        payment['pct'] = (payment['count'] / payment['total']) * 100
        payment = payment[payment['Churn'] == 'Yes']
        fig = px.bar(payment, y='PaymentMethod', x='pct', color='PaymentMethod', color_discrete_sequence=['#ef4444', '#06b6d4', '#06b6d4', '#06b6d4'], orientation='h')
        fig.update_layout(showlegend=False, xaxis_title='Churn (%)', yaxis_title='', xaxis=dict(range=[0,60]), margin=dict(t=0,b=0,l=0,r=0), height=210, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("""
        <h3 style="font-size:14px; margin:0 0 6px 0; color:#1e293b;font-weight:700;">Model Performance</h3>
    """, unsafe_allow_html=True)
    
    # Radio buttons for model selection
    selected_model = st.radio(
        "Select Model",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )
    
    # Get metrics for selected model
    m = model_metrics[selected_model]
    
    st.markdown(f"""
        <p style="margin:6px 0 8px 0;color:#64748b;font-size:12px;">{selected_model} Results</p>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
            <div style="background: white; border-left: 4px solid #3b82f6; background: linear-gradient(90deg, #f0f9ff 0%, white 100%); padding:14px;border-radius:16px;text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Accuracy</p>
                <p style="margin:6px 0 0 0;font-size:24px;font-weight:800;color:#1e293b;">{m['accuracy']:.1%}</p>
            </div>
            <div style="background: white; border-left: 4px solid #10b981; background: linear-gradient(90deg, #f0fdf4 0%, white 100%); padding:14px;border-radius:16px;text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Precision</p>
                <p style="margin:6px 0 0 0;font-size:24px;font-weight:800;color:#1e293b;">{m['precision']:.1%}</p>
            </div>
            <div style="background: white; border-left: 4px solid #f59e0b; background: linear-gradient(90deg, #fef7ed 0%, white 100%); padding:14px;border-radius:16px;text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Recall</p>
                <p style="margin:6px 0 0 0;font-size:24px;font-weight:800;color:#1e293b;">{m['recall']:.1%}</p>
            </div>
            <div style="background: white; border-left: 4px solid #8b5cf6; background: linear-gradient(90deg, #fdf4ff 0%, white 100%); padding:14px;border-radius:16px;text-align:center; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                <p style="margin:0;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;">ROC-AUC</p>
                <p style="margin:6px 0 0 0;font-size:24px;font-weight:800;color:#1e293b;">{m['roc_auc']:.1%}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
        <div class="card">
            <h3 style="font-size: 16px; margin: 0 0 16px 0; color: #1e293b; font-weight: 700;">Key Insights</h3>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:14px;">
                <div style="padding:14px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
                    <h4 style="font-size:14px;margin:0 0 6px 0;color:#1e293b;font-weight:700;">📊 Highest Churn</h4>
                    <p style="font-size:12px;color:#64748b;margin:0;line-height:1.5;">Month-to-month + Fiber optic + Electronic check</p>
                </div>
                <div style="padding:14px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
                    <h4 style="font-size:14px;margin:0 0 6px 0;color:#1e293b;font-weight:700;">🛡️ Best Retention</h4>
                    <p style="font-size:12px;color:#64748b;margin:0;line-height:1.5;">Two-year contract + Tech support + Auto payment</p>
                </div>
                <div style="padding:14px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
                    <h4 style="font-size:14px;margin:0 0 6px 0;color:#1e293b;font-weight:700;">⏱️ Critical Period</h4>
                    <p style="font-size:12px;color:#64748b;margin:0;line-height:1.5;">First 12 months - 40%+ churn risk</p>
                </div>
                <div style="padding:14px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
                    <h4 style="font-size:14px;margin:0 0 6px 0;color:#1e293b;font-weight:700;">💡 Opportunity</h4>
                    <p style="font-size:12px;color:#64748b;margin:0;line-height:1.5;">Offer discounts for 1-2 year contracts</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
