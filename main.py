import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import json
from predict_xgboost import predict_result


DF = pd.read_csv('QnA_Dataset.csv')
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
PERPLEXITY_KEY = "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"

def call_prompt(query):
    # Prepare payload for the Perplexity API
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-chat",
        "messages": [{'role': 'user', 'content': query}],
        "max_tokens": 100,
        "temperature": 0,
        "top_p": 0,
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
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
def get_perplexity_response(queries, topic):
    user_input = queries[-1]['content']
    top_5_relevant = get_top_answers(user_input)
    
    user_input =  'Relevant sample answer:\n' + ''.join([f'{idx}. {info}\n' for idx, info in enumerate(top_5_relevant)]) \
            + "\nPlease take reference from the sample answer above to answer the user ONLY when it is relevant. \nOtherwise just answer without the relevant answer's guidance.\n" \
            + f"""Please behave as a customer service person.
                Can you ask the user about this topic '{topic}' in the purpose of getting their data?

                Please also base your response on the user prompt:
                "{user_input}" """ 
             
            
    print(user_input)
    
    # perplexity_api_key = 
    
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
        "Authorization": f"Bearer {PERPLEXITY_KEY}",
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
    welcoming_response = "Welcome to Credit Risk Chatbot. May I know what is your current age?"
    # st.write(welcoming_response)
    st.session_state.messages.append({"role": "assistant", "content": welcoming_response})

if 'turn' not in st.session_state:
    st.session_state.turn = 0

if 'result' not in st.session_state:
    st.session_state.result = {'AGE': '', 'SEX': '', 'JOB and RESIDENCY':'', 'HOUSING':'', 'SAVING ACCOUNT BALANCE':'', 'CHECKING ACCOUNT BALANCE':'', 'CREDIT AMOUNT':'', 'CREDIT DURATION':'', 'PURPOSE OF CREDIT':'' }

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # print(st.session_state.messages)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Change this according to the validity 
current_topic_prompt = ['AGE', 'SEX', 'JOB and RESIDENCY', 'HOUSING',
                        'SAVING ACCOUNT BALANCE', 'CHECKING ACCOUNT BALANCE',
                        'CREDIT AMOUNT', 'CREDIT DURATION', 'PURPOSE OF CREDIT']
sample_output_information = {
    'AGE': 'Integer',
    'SEX': 'Integer: 1 for Male and 0 for Female', 
    'JOB and RESIDENCY': 'Integer: 0 for unskilled and non-resident, 1 for unskilled and resident, 2 for skilled, 3 for highly skilled.', 
    'HOUSING': 'String: "own", "rent", or "free".', 
    'SAVING ACCOUNT BALANCE': 'Integer. in HKD', 
    'CHECKING ACCOUNT BALANCE': 'Integer. in HKD', 
    'CREDIT AMOUNT': 'Integer. in HKD', 
    'CREDIT DURATION': 'Integer. in months', 
    'PURPOSE OF CREDIT': 'String. car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)' 
}
# numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled

# Extract result
def extract_result(user_prompt, topic):
    print("EXTRACT RESULT")
    res = call_prompt(f""" 
                      
Please extract the value of: '{topic}' based on the user response:
"{user_prompt}"

Return answer strictly as this JSON object: {{"result": (STRING containing extracted value. {sample_output_information[topic]})}}\n


Do not include anything else, just above's JSON object.
    """)
    
    print("Raw response:", res)  # Debugging line
    res = res.strip().replace('json', '').replace('`','').strip()
    try:
        # Load the response directly without modification
        res_json = json.loads(res)
        print("Parsed response: ", res_json)  # Print the parsed response
        print(res_json.get('result'))
        return res_json.get('result')  # Get validity
    except json.JSONDecodeError as e:
        print("FALSE!!!")
        print("JSON decode error:", e)
        return 0  # Handle the error appropriately
    

# Checking validity depending on the context
def check_validity(user_prompt, topic):
    print("CHECK VALID")
    res = call_prompt(f""" 
Does user answer according to the topic '{topic}' with correct data type? (e.g. Age would have integer, Name would have string answer and so forth, purpose of credit would have return string).

user_prompt:
'{user_prompt}'

Return answer strictly as this JSON object: {{"validity": (return 1 if True and return 0 if False)}}\n
Do not include anything else, just above's JSON object.
    """)

    print("Raw response:", res)  # Debugging line
    res = res.strip().replace('json', '').replace('`','').strip()
    try:
        # Load the response directly without modification
        res_json = json.loads(res)
        print("Parsed response: ", res_json)  # Print the parsed response
        print(res_json.get('validity'))
        return res_json.get('validity')  # Get validity
    except json.JSONDecodeError as e:
        print("FALSE!!!")
        print("JSON decode error:", e)
        return 0  # Handle the error appropriately

# Checking the topic turn
def check_turn(user_prompt, topic ,topic_before):
    print("CHECK TURN")
    res = call_prompt(f""" 
Please behave as a customer service person.

Can you ask the user about this topic of '{topic}' in the purpose of prompting them for their '{topic}'. 

Provide a transition sentence from the previous topic of '{topic_before}'.

Please also base your response on the user prompt:
"{user_prompt}"

Return answer strictly as this JSON object: e.g. {{"response": (STRING containing chatbot's response)}}\n
Do not include anything else, just above's JSON object.
    """)
    print("Raw response:", res)
    res = res.strip().replace('json', '').replace('`','').strip()
    try:
        # Load the response directly without modification
        res_json = json.loads(res)
        print("Parsed response: ", res_json)  # Print the parsed response
        print(res_json.get('response'))
        return res_json.get('response')  # Get validity
    except json.JSONDecodeError as e:
        print("FALSE!!!")
        print("JSON decode error:", e)
        return 0  # Handle the error appropriately


# Accept user input
if prompt := st.chat_input("Write down your prompt here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Check validity, it should prompt the user again if it is false
        valid = check_validity(prompt, current_topic_prompt[st.session_state.turn])
        print('valid: ', valid)
        
        # Not valid
        if not valid:  # Check if valid is False
            response = 'Invalid prompt, please try again.'
            st.write(response)
            # st.session_state.messages.append({"role": "assistant", "content": 'Invalid prompt, please try again.'})
            
        # Valid
        else:
            # Assign the extracted result to our result value
            st.session_state.result[current_topic_prompt[st.session_state.turn]] = extract_result(prompt, current_topic_prompt[st.session_state.turn])
            
            print("Result: ", st.session_state.result)
            
            pass_model = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # Let the turn to the next part
            st.session_state.turn += 1

            # If the session turn is over, meaning we have all of the data that we need.
            # Now we will call the prediction function and get the risk. 
            if st.session_state.turn >= len(current_topic_prompt):
                # Call the model
                risk = predict_result(st.session_state.result)[0]
                risk_response = "good risk"
                if risk == 1:
                    risk_response = "bad risk"
                credit_score = predict_result(st.session_state.result)[1]
                response = f"You have a {risk_response}, your credit risk score is: {credit_score}"
                st.write(response)
            
            else:
                # Make the response according to the current turn current_topic_prompt[st.session_state.turn]
                response = check_turn(prompt, current_topic_prompt[st.session_state.turn], current_topic_prompt[st.session_state.turn - 1]) #get_perplexity_response(pass_model, st.session_state.turn ) 
                #check_turn(prompt, current_topic_prompt[st.session_state.turn]) ##get_perplexity_response(pass_model)
                print("Answer:", response)
                
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
