import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ABC Ltd | Late Delivery Risk", page_icon="📦", layout="centered")


@st.cache_resource
def load_models():
    return joblib.load("late_logit.joblib"), joblib.load("days_linear.joblib")


logit, lin = load_models()

# Promised days per shipping mode (from the data: each mode has one fixed value)
SCHEDULED = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}
MARKETS = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]
TYPES = ["CASH", "DEBIT", "PAYMENT", "TRANSFER"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
# 'Health and Beauty ' has a trailing space in the training data - keep it exactly
DEPARTMENTS = ["Apparel", "Book Shop", "Discs Shop", "Fan Shop", "Fitness", "Footwear",
               "Golf", "Health and Beauty ", "Outdoors", "Pet Shop", "Technology"]

LOW, HIGH = 0.40, 0.60


def make_row(mode, market, ptype, segment, dept, qty, discount, sales):
    return pd.DataFrame([{
        "Shipping Mode": mode,
        "Market": market,
        "Type": ptype,
        "Customer Segment": segment,
        "Department Name": dept,
        "Order Item Quantity": qty,
        "Order Item Discount Rate": discount,
        "Sales": sales,
    }])


def band(p):
    if p < LOW:
        return "Low", "success", "Standard handling is fine. No action needed."
    if p < HIGH:
        return "Medium", "warning", "Keep an eye on this order and consider a proactive customer update."
    return "High", "error", ("Consider expediting, switching to a faster mode, "
                             "or warning the customer before dispatch.")


st.title("📦 Late Delivery Risk Checker")
st.caption("ABC Ltd | Enter order details before dispatch to see the risk of late delivery.")

with st.form("order"):
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Shipping mode", list(SCHEDULED.keys()), index=3)
    market = c2.selectbox("Market", MARKETS)
    ptype = c1.selectbox("Payment type", TYPES)
    segment = c2.selectbox("Customer segment", SEGMENTS)
    dept = c1.selectbox("Department", DEPARTMENTS, format_func=str.strip)
    qty = c2.slider("Quantity", 1, 5, 1)
    discount_pct = c1.slider("Discount (%)", 0, 25, 10)
    sales = c2.number_input("Order value ($)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    go = st.form_submit_button("Check risk", use_container_width=True)

if go:
    row = make_row(mode, market, ptype, segment, dept, qty, discount_pct / 100, sales)
    prob = float(logit.predict_proba(row)[0, 1])
    days = max(0.0, float(lin.predict(row)[0]))
    label, kind, advice = band(prob)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of being late", f"{prob:.0%}")
    m2.metric("Risk level", label)
    m3.metric("Expected days vs promised", f"{days:.1f} vs {SCHEDULED[mode]}")
    getattr(st, kind)(f"**{label} risk.** {advice}")

    st.subheader("What if you changed the shipping mode?")
    rows = []
    for m in SCHEDULED:
        r = make_row(m, market, ptype, segment, dept, qty, discount_pct / 100, sales)
        rows.append({
            "Shipping mode": m,
            "Promised days": SCHEDULED[m],
            "Expected days": round(max(0.0, float(lin.predict(r)[0])), 1),
            "Chance of being late": f"{float(logit.predict_proba(r)[0, 1]):.0%}",
            "Risk level": band(float(logit.predict_proba(r)[0, 1]))[0],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("How does this prediction work? How far can I trust it?"):
    st.markdown(
        """
- The tool uses two statistical models trained on about 180,000 past ABC Ltd orders:
  one estimates the **chance of lateness**, the other the **expected shipping days**.
- **Shipping mode is by far the biggest factor.** Market, payment type, segment, product
  department, quantity, discount and order value change the result only slightly.
- On orders the model had not seen, it was **correct about 7 in 10 times**. It is a
  decision aid, not a guarantee. Combine it with your own knowledge of the order.
- Risk levels: **Low** below 40%, **Medium** 40 to 60%, **High** above 60%.
- The data does not include weather, carrier delays, stock levels or holidays, which
  can also cause late deliveries.
        """
    )
