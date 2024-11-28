import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss

DF = pd.read_csv('QnA_Dataset.csv')
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Function to encode documents in batches
def batch_encode(batch_size=32):
    all_embeddings = []
    for item in np.array(DF['Question']):
        
        # Encode the batch and append to the list
        batch_embeddings = MODEL.encode(item, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings into a single numpy array
    return np.vstack(all_embeddings)

# Call the function
embeddings = batch_encode()

index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
index.add(embeddings)  # Add embeddings to the index


def get_top_answers(question, top_n=5):
    user_embedding = MODEL.encode([question], convert_to_numpy=True)
    # Retrieve the most similar questions from the index
    D, I = index.search(user_embedding, k=top_n)  # Fetch top N results
    print(D)
    print(I)
    relevant_docs = [DF.loc[idx, 'Answer' ] for idx in I[0]]
    
    return relevant_docs

# Function to get a response from the Perplexity API
def get_perplexity_response(queries):
    user_input = queries[-1]['content']
    top_5_relevant = get_top_answers(user_input)
    
    user_input =  'Relevant sample answer:\n' + ''.join([f'{idx}. {info}\n' for idx, info in enumerate(top_5_relevant)]) \
            + "\nPlease take reference from the sample answer above to answer the user ONLY when it is relevant. \nOtherwise just answer without the relevant answer's guidance.\n" \
            + 'user_input: ' + user_input 
    print(user_input)
    
    perplexity_api_key = "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"
    
    queries[-1]['content'] = user_input

    print(queries)
    
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
if prompt := st.chat_input("Write down your prompt here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
# Display assistant response in chat message container
    with st.chat_message("assistant"):

        pass_model = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        # print("pass_model: " ,pass_model)
        stream = get_perplexity_response(pass_model)
        print("Answer:", stream)
        response = st.write(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})