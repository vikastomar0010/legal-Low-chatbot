import streamlit as st
import requests

st.title("⚖️ Legal Chatbot")

backend_url = "http://localhost:8000/chat"

query = st.text_input("Ask your legal question:")
city = st.text_input("Enter your city (optional):")

if st.button("Get Response"):
    payload = {"query": query, "city": city}
    response = requests.post(backend_url, json=payload)

    if response.status_code == 200:
        data = response.json()
        st.subheader("🔹 Legal Context")
        st.write(data['context'])

        st.subheader("🔹 Legal Advice")
        st.write(data['advice'])

        if data.get("location"):
            st.subheader("📍 Nearby Legal Centers")
            for place in data["location"]:
                st.write(f"- {place}")
    else:
        st.error("Backend not responding!")
