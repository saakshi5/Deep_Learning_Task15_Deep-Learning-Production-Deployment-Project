
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="IT Incident AI",
    page_icon="🛠️",
    layout="centered",
)

st.title("🛠️ IT Incident Classification System")

st.write(
    "Enter an IT support incident description to predict "
    "its likely category using a deep learning model."
)

API_URL = os.getenv(
    "API_URL", "http://127.0.0.1:5000"
).rstrip("/")

EXAMPLES = [
    "My laptop cannot connect to office Wi-Fi",
    "I received a suspicious email asking for my password",
    "Please unlock my employee account",
    "The application crashes when I open it",
    "My monitor is flickering and going black",
]

with st.sidebar:
    st.subheader("API service")

    st.write(f"API address: {API_URL}")

    if st.button("Check API health"):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )

            if response.ok:
                st.success("API is healthy")
                st.json(response.json())
            else:
                st.error(
                    f"API returned HTTP {response.status_code}"
                )

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

    st.caption("Educational demonstration application.")

selected = st.selectbox(
    "Choose an example or enter your own incident",
    ["Choose an example"] + EXAMPLES,
)

default_text = (
    "" if selected == "Choose an example" else selected
)

incident = st.text_area(
    "Incident description",
    value=default_text,
    height=130,
    max_chars=2000,
    placeholder="Describe the IT issue here...",
)

if st.button(
    "Classify Incident",
    type="primary",
    use_container_width=True,
):
    if not incident.strip():
        st.warning("Please enter an incident description.")

    else:
        try:
            with st.spinner("Classifying incident..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": incident},
                    timeout=30,
                )

            if response.ok:
                result = response.json()

                st.success(
                    "Predicted category: "
                    + result["predicted_category"]
                )

                st.metric(
                    "Model confidence",
                    f"{result['confidence']:.1%}",
                )

                st.subheader("Category confidence scores")

                chart_data = {
                    item["category"]: item["confidence"]
                    for item in result["probabilities"]
                }

                st.bar_chart(chart_data)

                st.caption(
                    "Confidence is the model's output score, "
                    "not a guarantee of correctness."
                )

            else:
                st.error(
                    f"API error {response.status_code}: "
                    f"{response.text}"
                )

        except requests.RequestException as exc:
            st.error(
                "Unable to connect to Flask. Check that "
                f"the API is running. Details: {exc}"
            )

st.divider()

st.caption(
    "Do not enter passwords or confidential information. "
    "Review predictions before taking action."
)
