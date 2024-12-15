import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import time
import os

load_dotenv()
api_key = os.getenv("API_KEY")

def stream_data(response_text):
    for word in response_text.split(" "):
        yield word + " "
        time.sleep(0.02)


def main():
    # Configure the GenAI API
    genai.configure(api_key=api_key)
    gen_ai_model = genai.GenerativeModel("gemini-1.5-flash")

    # Set up chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

        # Add the initial greeting as the first message
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hello 👋, what would you like to know in regards to finances and loans?"})

    # Display chat messages from the session state
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box for user prompt
    prompt = st.chat_input("Enter your question here:")

    if prompt:
        # Append user's message to the session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the assistant's response
        with st.chat_message("assistant"):
            response = gen_ai_model.generate_content(
                f"""
                You are a trained finance loan expert.
                If asked anything outside the scope, respond with 'Sorry, can't help you with that.'
                Your response should not be more than 100 words or less than 30 words.
                Give relevant output only.

                My questions are: {prompt}
                """
            )
            assistant_message = response.text

            # Stream the assistant's message
            st.write_stream(stream_data(assistant_message))

            # Append assistant's message to the session state
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})


if __name__ == '__main__':
    st.set_page_config(
        page_title="Chat Section",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    main()



