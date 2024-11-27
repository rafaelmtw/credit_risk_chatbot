import streamlit as st
import requests
# [
#             {"role": "system", "content": "Keep response within 50 words \n Avoid the use of symbols such as ( ) * & ^ % $ # @ !. Response concisely. "},
#             {"role": "user", "content": f"User input: {user_input}."}
#         ]

# Function to get a response from the Perplexity API
def get_perplexity_response(queries):
    
    perplexity_api_key = "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"

    # Prepare payload for the Perplexity API
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-chat",
        "messages": queries,
        "max_tokens": 100,
        "temperature": 0,
        "top_p": 0,
    }

    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
    
        return assistant_message
    
    except requests.exceptions.RequestException as e:
        print(f"ERROR:\n {str(e)}")

        
# Streamlit UI
st.title("Chatbot with Credit Risk API")

# Display chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
# Display assistant response in chat message container
    with st.chat_message("assistant"):
        # stream = client.chat.completions.create(
        #     model=st.session_state["openai_model"],
        #     messages=[
        #         {"role": m["role"], "content": m["content"]}
        #         for m in st.session_state.messages
        #     ],
        #     stream=True,
        # )
        pass_model = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        print("pass_model: " ,pass_model)
        stream = get_perplexity_response(pass_model)
        print("stream:", stream)
        response = st.write(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})