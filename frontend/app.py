import streamlit as st
from rag_module import get_answer   # import your function directly

st.title("⚖️ Legal Chatbot")

query = st.text_input("Ask your legal question:")
city = st.text_input("Enter your city (optional):")

if st.button("Get Response"):

    if not query:
        st.warning("Please enter a legal question.")
    else:
        with st.spinner("Analyzing your legal query..."):
            data = get_answer(query, city)

        st.subheader("🔹 Legal Context")
        st.write(data["context"])

        st.subheader("🔹 Legal Advice")
        st.write(data["advice"])

        if data.get("location"):
            st.subheader("📍 Nearby Legal Centers")
            for place in data["location"]:
                st.write(f"- {place}")
