import streamlit as st
import requests

st.set_page_config(page_title="Web-Connected Streamlit App", layout="centered")

# Title
st.title("🌐 Web-Connected Demo App")

# Instructions
st.write("""
This Streamlit app connects to the internet to fetch data from a public API. 
Share your deployed link with others to let them access it from anywhere.
""")

# API Selection
api_options = {
    "Public IP Address": "https://api.ipify.org?format=json",
    "Random Joke": "https://official-joke-api.appspot.com/random_joke",
    "Current Bitcoin Price (USD)": "https://api.coindesk.com/v1/bpi/currentprice.json"
}
api_choice = st.selectbox("Choose a public API to query:", list(api_options.keys()))

# Button
if st.button("Fetch Data"):
    api_url = api_options[api_choice]
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # Display data based on selected API
        if api_choice == "Public IP Address":
            st.success(f"Your Public IP Address is: {data['ip']}")
        elif api_choice == "Random Joke":
            st.write(f"**{data['setup']}**")
            st.write(f"👉 {data['punchline']}")
        elif api_choice == "Current Bitcoin Price (USD)":
            price = data["bpi"]["USD"]["rate"]
            st.metric(label="💰 Bitcoin Price (USD)", value=f"${price}")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")

# Footer
st.markdown("---")
st.caption("Built with gpt using Streamlit and public APIs")