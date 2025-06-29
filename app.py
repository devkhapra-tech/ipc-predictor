import streamlit as st
from main import predict
st.title('VERDICTO- IPC PREDICTION SYSTEM')


user_input = st.text_area("Describe the crime", placeholder="e.g., Theft in a house using a knife in Delhi")
top_n = st.slider("How many IPC sections to show?", 1, 10, 5)

if st.button("Predict"):
    if user_input.strip():
        results = predict(user_input, top_n)
        st.success(f"Top {top_n} Relevant IPC Sections:")
        for i, res in enumerate(results, 1):
            st.markdown(f"**{i}. {res['section']}**  \nSimilarity Score: `{res['score']:.4f}`")
    else:
        st.warning("Please enter a crime description.")
