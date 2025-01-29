from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import streamlit as st
import os
from sqlalchemy import create_engine, text
import uuid

# Set up Streamlit page
st.title("🚀 Deepseek-R1 Llama-70B Chat")
st.write("⚡ Blazing-fast Thinking Model powered by Groq ❤️")

# Add sidebar for API key input and details
with st.sidebar:
    st.header("⚙️ Configuration")
    # Use secrets for API keys and database URL
    st.session_state.groq_api_key = st.secrets["GROQ_API_KEY"]
    st.session_state.neon_database_url = st.secrets["NEON_DATABASE_URL"]
    
    # Add optional chat_id field
    st.session_state.chat_id = st.text_input("Chat ID (optional)", key="chat_id_input")
    
    # Add model details section
    st.divider()
    st.markdown("**Model Details**")
    st.caption("Running: `deepseek-r1-distill-llama-70b`")
    st.caption("Groq LPU Inference Engine")
    
    # Add New Chat button
    st.divider()
    if st.button("🔄 Start New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())  # Generate new session ID
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant.")
        ]
        st.rerun()

# Initialize session ID if not exists
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to save message to database
def save_message_to_db(session_id, user_message, ai_message, thinking_process=None, chat_id=None):
    if st.session_state.neon_database_url:
        try:
            engine = create_engine(st.session_state.neon_database_url)
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO chat_history (session_id, chat_id, user_message, thinking_process, ai_message)
                    VALUES (:session_id, :chat_id, :user_message, :thinking_process, :ai_message)
                """), {
                    "session_id": session_id,
                    "chat_id": chat_id,
                    "user_message": user_message,
                    "thinking_process": thinking_process,
                    "ai_message": ai_message
                })
                conn.commit()
        except Exception as e:
            st.error(f"Failed to save to database: {str(e)}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]

# Display welcome message
with st.chat_message("assistant"):
    st.write("Ask me anything!")

# Display chat history
for message in st.session_state.messages[1:]:  # Skip the system message
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            content = message.content
            if "<think>" in content and "</think>" in content:
                think_start = content.index("<think>")
                think_end = content.index("</think>") + len("</think>")
                thinking = content[think_start:think_end]
                actual_response = content[think_end:].strip()
                
                with st.expander("Show AI thinking process"):
                    st.write(thinking)
                st.write(actual_response)
            else:
                st.write(content)

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    if not st.session_state.groq_api_key:
        st.error("Please enter your GROQ API key in the sidebar")
        st.stop()
        
    # Initialize the ChatOpenAI model with Groq
    chat = ChatOpenAI(
        model="deepseek-r1-distill-llama-70b",
        openai_api_key=st.session_state.groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1"
    )
    
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        message_placeholder = st.empty()
        full_response = ""
        
        for chunk in chat.stream(st.session_state.messages):
            if chunk.content:
                full_response += chunk.content
                if "<think>" not in full_response:
                    message_placeholder.write(full_response)
        
        # Process thinking tags and response
        thinking_process = None
        ai_message = full_response
        
        if "<think>" in full_response and "</think>" in full_response:
            think_start = full_response.index("<think>")
            think_end = full_response.index("</think>") + len("</think>")
            thinking_process = full_response[think_start:think_end]
            ai_message = full_response[think_end:].strip()
            
            with thinking_placeholder:
                with st.expander("Show AI thinking process"):
                    st.write(thinking_process)
            message_placeholder.write(ai_message)
        else:
            thinking_placeholder.empty()
            message_placeholder.write(ai_message)
    
    # Save to database
    save_message_to_db(
        session_id=st.session_state.session_id,
        user_message=prompt,
        ai_message=ai_message,
        thinking_process=thinking_process,
        chat_id=st.session_state.chat_id if st.session_state.chat_id else None
    )
    
    # Add AI response to chat history
    st.session_state.messages.append(AIMessage(content=full_response))
