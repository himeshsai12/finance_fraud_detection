"""Streamlit demo for the real-time fraud detection model."""

from __future__ import annotations

import streamlit as st

from fraud_detection.service import TransactionRequest, score_transaction

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #07111b;
        color: #edf3f8;
    }
    [data-testid="stSidebar"] {
        background: rgba(17, 24, 39, 0.9);
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        margin-bottom: 0.25rem !important;
    }
    h3 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
    }
    .stMetric { 
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        padding: 1rem 1.1rem;
    }
    .stMetric label {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 3.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: #f8fafc !important;
    }
    div[data-testid="stAlert"] {
        background: rgba(132, 132, 62, 0.45);
        border: 1px solid rgba(198, 172, 84, 0.55);
        color: #f8fafc;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        font-size: 1.1rem;
    }
    .stForm {
        background: rgba(8, 15, 23, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 1.25rem;
    }
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.9);
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 10px;
    }
    button[kind="primary"] {
        border-radius: 10px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }
    .sidebar-content {
        padding-top: 1.2rem;
    }
    .sidebar-content h2 {
        color: #f8fafc !important;
        font-size: 1.2rem !important;
    }
    .sidebar-content p {
        color: #dbeafe !important;
        font-size: 1.05rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Real-time Fraud Detection Dashboard")
st.caption("Evaluate a transaction risk score with the project’s fraud scoring logic.")

with st.form("fraud_score_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Amount",
            min_value=0.01,
            value=5000.0,
            step=100.0,
            format="%.2f",
        )
        transaction_type = st.selectbox(
            "Transaction type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT", "OTHER"],
            index=1,
        )
        name_orig = st.text_input("Origin account", value="customer_a")

    with col2:
        name_dest = st.text_input("Destination account", value="merchant_b")
        old_balance_org = st.number_input(
            "Old balance (origin)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )
        new_balance_dest = st.number_input(
            "New balance (destination)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
        )

    submitted = st.form_submit_button("Score transaction")

if submitted:
    request = TransactionRequest(
        amount=amount,
        type=transaction_type,
        nameOrig=name_orig,
        nameDest=name_dest,
        oldbalanceOrg=old_balance_org,
        newbalanceDest=new_balance_dest,
    )
    score = score_transaction(request)

    st.subheader("Risk result")

    score_col, delta_col = st.columns([2, 1])
    with score_col:
        st.metric("Fraud risk score", f"{score:.4f}", delta="0-1 range")

    if score < 0.2:
        st.success("Low risk: this transaction appears normal for the current heuristic model.")
    elif score < 0.6:
        st.info("Moderate risk: additional manual review is recommended.")
    else:
        st.warning("High risk: this transaction is flagged for investigation.")

    st.json(
        {
            "status": "ok",
            "score": score,
            "type": request.type,
            "amount": request.amount,
            "nameOrig": request.nameOrig,
            "nameDest": request.nameDest,
        }
    )

st.sidebar.header("About")
st.sidebar.write(
    "This demo uses the same fraud heuristic that powers the FastAPI scoring endpoint in the project."
)
st.sidebar.write("It is intended for local exploration and demos.")
