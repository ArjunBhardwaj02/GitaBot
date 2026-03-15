import streamlit as st
from backend import app 
from langchain_core.messages import HumanMessage

# --- CONFIGURATION ---
st.set_page_config(page_title="GitaBot AI", page_icon="🪷", layout="centered")

# --- CUSTOM CSS FOR HOMEPAGE AESTHETIC ---
st.markdown("""
    <style>
        /* Center all elements */
        .stApp {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        /* Style the 'Radhe Radhe' heading */
        .radhe-header {
            font-size: 70px !important;
            font-weight: 800;
            color: #FF9933; /* A spiritual orange color */
            text-align: center;
            margin-top: 100px; /* Push it down the page */
            margin-bottom: 10px;
        }
        
        /* Style the sub-heading */
        .sub-header {
            font-size: 32px !important;
            color: #5f6368; /* Subtle grey like Gemini */
            text-align: center;
            margin-bottom: 50px;
        }

        /* Fix the chat input to the bottom */
        .stChatInput {
            position: fixed;
            bottom: 30px;
            width: 100%;
            max-width: 700px;
        }
    </style>
""", unsafe_allow_html=True)


# --- HOMEPAGE INITIAL STATE ---
# Only show the big greeting if there are no messages yet
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.markdown('<p class="radhe-header">Radhe Radhe</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">What is on your mind?</p>', unsafe_allow_html=True)


# --- CHAT LOGIC ---
# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Persistent thread ID for the web session
chat_config = {"configurable": {"thread_id": "streamlit_web_user_session"}}

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask for spiritual guidance..."):
    # Clear the homepage greeting by adding the first message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Rerun the script to clear the big headers and show the chat bubble
    st.rerun()

# This part executes after st.rerun() if there is input
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    # Display assistant response bubble
    with st.chat_message("assistant"):
        with st.spinner("Consulting the scriptures..."):
            # Call your LangGraph backend
            response = app.invoke({"question": user_prompt, "generation":[HumanMessage(content=user_prompt)]}, config=chat_config)
            
            # Extract final answer
            final_answer = response['generation'][-1].content
            st.markdown(final_answer)
            
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": final_answer})