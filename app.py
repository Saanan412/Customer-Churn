import streamlit as st
import pandas as pd
import joblib
import time


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telecom Customer Churn",
    page_icon="📱",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("customer_churn_scaler.pkl")
columns = joblib.load("customer_churn_columns.pkl")


# =========================================================
# CSS + HTML
# =========================================================

st.html("""
<style>

.stApp {
    background:
        linear-gradient(
            135deg,
            #0f172a,
            #172554,
            #312e81
        );

    color: white;
}


/* MAIN WIDTH */

.block-container {
    max-width: 1200px;
    padding-top: 30px;
}


/* HEADER */

.header {
    text-align: center;
    padding: 40px 25px;
    margin-bottom: 35px;

    border-radius: 28px;

    background:
        linear-gradient(
            270deg,
            #4f46e5,
            #7c3aed,
            #2563eb,
            #4f46e5
        );

    background-size: 400% 400%;

    animation: gradientMove 8s ease infinite;

    box-shadow:
        0 15px 45px rgba(0,0,0,0.4);
}

.header h1 {
    color: white;
    font-size: 44px;
    margin: 0;

    animation: fadeDown 1s ease;
}

.header p {
    color: #e0e7ff;
    font-size: 18px;
    margin-top: 12px;
}


/* SECTION TITLE */

.section-title {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin: 30px 0;
}


/* INPUT CARD */

.input-card {
    background: rgba(255,255,255,0.09);

    border:
        1px solid rgba(255,255,255,0.18);

    border-radius: 20px;

    padding: 20px;

    min-height: 120px;

    margin-bottom: 8px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.25);

    transition: 0.3s;
}

.input-card:hover {
    transform: translateY(-5px);

    background:
        rgba(255,255,255,0.14);

    box-shadow:
        0 15px 35px rgba(0,0,0,0.35);
}

.input-card h3 {
    margin: 0;
    color: white;
    font-size: 19px;
}

.input-card p {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 10px;
}


/* INPUTS */

.stNumberInput input {
    background: rgba(255,255,255,0.08);
    color: white;

    border:
        1px solid rgba(255,255,255,0.2);

    border-radius: 12px;
}


.stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08);
    color: white;

    border:
        1px solid rgba(255,255,255,0.2);

    border-radius: 12px;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    height: 60px;

    border: none;
    border-radius: 16px;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-size: 20px;
    font-weight: bold;

    transition: 0.3s;

    box-shadow:
        0 10px 30px rgba(37,99,235,0.4);
}

.stButton > button:hover {
    transform: scale(1.02);

    box-shadow:
        0 15px 40px rgba(124,58,237,0.6);
}


/* LOADING */

.loader {
    text-align: center;

    padding: 30px;

    font-size: 24px;

    animation: pulse 1s infinite;
}


/* RESULT */

.result-card {
    text-align: center;

    margin-top: 35px;

    padding: 40px;

    border-radius: 25px;

    background:
        rgba(255,255,255,0.09);

    border:
        1px solid rgba(255,255,255,0.18);

    box-shadow:
        0 15px 45px rgba(0,0,0,0.35);

    animation: resultAppear 0.7s ease;
}


.result-title {
    font-size: 32px;
    font-weight: bold;
    color: white;
}


/* PROBABILITY */

.probability {
    font-size: 50px;
    font-weight: bold;
    color: white;

    margin: 20px 0;
}


/* PROGRESS */

.progress-container {
    width: 100%;
    height: 28px;

    background:
        rgba(255,255,255,0.15);

    border-radius: 50px;

    overflow: hidden;

    margin: 20px 0;
}

.progress-bar {
    height: 100%;

    border-radius: 50px;

    background:
        linear-gradient(
            90deg,
            #22c55e,
            #eab308,
            #ef4444
        );

    animation:
        progressAnimation 1.5s ease;
}


/* RISK */

.risk {
    display: inline-block;

    padding: 12px 30px;

    border-radius: 50px;

    font-size: 20px;

    font-weight: bold;

    margin: 15px;
}

.low {
    background: rgba(34,197,94,0.2);
    border: 1px solid #22c55e;
    color: #86efac;
}

.medium {
    background: rgba(234,179,8,0.2);
    border: 1px solid #eab308;
    color: #fde047;
}

.high {
    background: rgba(239,68,68,0.2);
    border: 1px solid #ef4444;
    color: #fca5a5;

    animation: warningPulse 1.2s infinite;
}


/* ANIMATIONS */

@keyframes gradientMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}


@keyframes fadeDown {

    from {
        opacity: 0;
        transform: translateY(-30px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes resultAppear {

    from {
        opacity: 0;
        transform: scale(0.85);
    }

    to {
        opacity: 1;
        transform: scale(1);
    }
}


@keyframes pulse {

    0% {
        transform: scale(1);
    }

    50% {
        transform: scale(1.05);
    }

    100% {
        transform: scale(1);
    }
}


@keyframes warningPulse {

    0% {
        box-shadow: 0 0 0 rgba(239,68,68,0.2);
    }

    50% {
        box-shadow: 0 0 30px rgba(239,68,68,0.8);
    }

    100% {
        box-shadow: 0 0 0 rgba(239,68,68,0.2);
    }
}


@keyframes progressAnimation {

    from {
        width: 0%;
    }
}


/* MOBILE */

@media (max-width: 768px) {

    .header h1 {
        font-size: 30px;
    }

    .header p {
        font-size: 15px;
    }

    .result-title {
        font-size: 25px;
    }

    .probability {
        font-size: 38px;
    }

}

</style>
""")


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="header">

    <h1>📱 Telecom Customer Churn Prediction</h1>

    <p>
        AI-powered customer churn prediction system
    </p>

</div>
""")


st.write(
    "Enter the customer's information below "
    "to predict whether the customer is likely to churn."
)


# =========================================================
# SECTION TITLE
# =========================================================

st.html("""
<div class="section-title">
    💡 Customer Information
</div>
""")


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.html("""
    <div class="input-card">

        <h3>📅 Account Weeks</h3>

        <p>
            Number of weeks the customer has had an account.
        </p>

    </div>
    """)

    account_weeks = st.number_input(
        "Account Weeks",
        min_value=1,
        max_value=243,
        value=101,
        step=1
    )


with col2:

    st.html("""
    <div class="input-card">

        <h3>🔄 Contract Renewal</h3>

        <p>
            Whether the customer has renewed the contract.
        </p>

    </div>
    """)

    contract_renewal = st.selectbox(
        "Contract Renewal",
        [0, 1],
        format_func=lambda x:
            "❌ No" if x == 0 else "✅ Yes"
    )


with col3:

    st.html("""
    <div class="input-card">

        <h3>📶 Data Plan</h3>

        <p>
            Whether the customer has a data plan.
        </p>

    </div>
    """)

    data_plan = st.selectbox(
        "Data Plan",
        [0, 1],
        format_func=lambda x:
            "❌ No Data Plan"
            if x == 0
            else "✅ Has Data Plan"
    )


# =========================================================
# ROW 2
# =========================================================

col4, col5, col6 = st.columns(3)


with col4:

    st.html("""
    <div class="input-card">

        <h3>📊 Data Usage</h3>

        <p>
            Customer's data usage.
        </p>

    </div>
    """)

    data_usage = st.number_input(
        "Data Usage",
        min_value=0.0,
        max_value=5.4,
        value=0.8,
        step=0.1
    )


with col5:

    st.html("""
    <div class="input-card">

        <h3>☎️ Customer Service Calls</h3>

        <p>
            Number of calls made to customer service.
        </p>

    </div>
    """)

    cust_serv_calls = st.number_input(
        "Customer Service Calls",
        min_value=0,
        max_value=9,
        value=1,
        step=1
    )


with col6:

    st.html("""
    <div class="input-card">

        <h3>📞 Day Minutes</h3>

        <p>
            Number of minutes used during the day.
        </p>

    </div>
    """)

    day_mins = st.number_input(
        "Day Minutes",
        min_value=0.0,
        max_value=350.8,
        value=179.8,
        step=0.1
    )


# =========================================================
# ROW 3
# =========================================================

col7, col8, col9 = st.columns(3)


with col7:

    st.html("""
    <div class="input-card">

        <h3>📱 Day Calls</h3>

        <p>
            Number of calls made during the day.
        </p>

    </div>
    """)

    day_calls = st.number_input(
        "Day Calls",
        min_value=0,
        max_value=165,
        value=100,
        step=1
    )


with col8:

    st.html("""
    <div class="input-card">

        <h3>💰 Monthly Charge</h3>

        <p>
            Customer's monthly service charge.
        </p>

    </div>
    """)

    monthly_charge = st.number_input(
        "Monthly Charge",
        min_value=14.0,
        max_value=111.3,
        value=56.3,
        step=0.1
    )


with col9:

    st.html("""
    <div class="input-card">

        <h3>💸 Overage Fee</h3>

        <p>
            Additional charges due to usage overage.
        </p>

    </div>
    """)

    overage_fee = st.number_input(
        "Overage Fee",
        min_value=0.0,
        max_value=18.19,
        value=10.05,
        step=0.01
    )


# =========================================================
# ROW 4
# =========================================================

col10, col11, col12 = st.columns(3)


with col10:

    st.html("""
    <div class="input-card">

        <h3>🌍 Roaming Minutes</h3>

        <p>
            Number of minutes used while roaming.
        </p>

    </div>
    """)

    roam_mins = st.number_input(
        "Roaming Minutes",
        min_value=0.0,
        max_value=20.0,
        value=10.2,
        step=0.1
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")
st.write("")

predict = st.button(
    "🔮 Predict Customer Churn"
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # LOADING
    # -----------------------------------------------------

    loading = st.empty()

    loading.html("""
    <div class="loader">

        🔄 Analyzing customer information...

        <br>

        <small>
            AI model is processing the input...
        </small>

    </div>
    """)

    time.sleep(1.5)

    loading.empty()


    # -----------------------------------------------------
    # CREATE INPUT
    # -----------------------------------------------------

    raw_input = {

        "AccountWeeks": account_weeks,

        "ContractRenewal": contract_renewal,

        "DataPlan": data_plan,

        "DataUsage": data_usage,

        "CustServCalls": cust_serv_calls,

        "DayMins": day_mins,

        "DayCalls": day_calls,

        "MonthlyCharge": monthly_charge,

        "OverageFee": overage_fee,

        "RoamMins": roam_mins
    }


    # -----------------------------------------------------
    # DATAFRAME
    # -----------------------------------------------------

    input_df = pd.DataFrame([raw_input])


    # -----------------------------------------------------
    # MATCH MODEL COLUMNS
    # -----------------------------------------------------

    for col in columns:

        if col not in input_df.columns:

            input_df[col] = 0


    input_df = input_df[columns]


    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    scaled_input = scaler.transform(input_df)


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    prediction = model.predict(
        scaled_input
    )[0]


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            scaled_input
        )[0]

        churn_probability = (
            probabilities[1] * 100
        )

    else:

        churn_probability = 0


    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    if churn_probability < 30:

        risk = "LOW RISK"
        risk_class = "low"
        risk_icon = "🎉"

    elif churn_probability < 70:

        risk = "MEDIUM RISK"
        risk_class = "medium"
        risk_icon = "⚡"

    else:

        risk = "HIGH RISK"
        risk_class = "high"
        risk_icon = "⚠️"


    # =====================================================
    # RESULT
    # =====================================================

    st.html("""
    <div class="result-card">
    """)


    if prediction == 1:

        st.html("""
        <div class="result-title">
            ⚠️ Customer is Likely to Churn
        </div>
        """)

        st.warning(
            "The model predicts that this customer "
            "may churn."
        )

    else:

        st.html("""
        <div class="result-title">
            🎉 Customer is Unlikely to Churn
        </div>
        """)

        st.success(
            "The model predicts that this customer "
            "is unlikely to churn."
        )

        st.balloons()


    # =====================================================
    # RISK
    # =====================================================

    st.html(f"""
    <div class="risk {risk_class}">

        {risk_icon} {risk}

    </div>
    """)


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.html(f"""
    <div class="probability">

        🎯 {churn_probability:.2f}%

    </div>

    <p>
        Churn Probability
    </p>

    <div class="progress-container">

        <div
            class="progress-bar"
            style="width: {churn_probability}%;">
        </div>

    </div>
    """)


    # =====================================================
    # RISK MESSAGE
    # =====================================================

    if churn_probability < 30:

        st.info(
            "💡 Low Churn Risk — "
            "The predicted probability of churn is low."
        )

    elif churn_probability < 70:

        st.warning(
            "💡 Medium Churn Risk — "
            "The predicted probability of churn is moderate."
        )

    else:

        st.error(
            "💡 High Churn Risk — "
            "The predicted probability of churn is high."
        )


    # CLOSE RESULT CARD

    st.html("""
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div style="
    text-align:center;
    margin-top:50px;
    padding:25px;
    color:#cbd5e1;
">

    📊 Telecom Customer Churn Prediction

    <br><br>

    Built with Python • Scikit-learn • Streamlit

</div>
""")