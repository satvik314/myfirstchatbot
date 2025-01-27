from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import streamlit as st
import os
from sqlalchemy import create_engine, text
import uuid
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory

# Set up Streamlit page
st.title("🚀 Deepseek-R1 Llama-70B Chat")
st.write("⚡ Blazing-fast Thinking Model powered by Groq ❤️")

# Add sidebar for API key input and details
with st.sidebar:
    st.header("⚙️ Configuration")
    # Use secrets directly instead of text inputs
    st.session_state.groq_api_key = st.secrets["GROQ_API_KEY"]
    st.session_state.neon_database_url = st.secrets["NEON_DATABASE_URL"]
    
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
    
    # Add branding with hyperlink
    st.divider()
    st.markdown(
        "**Built by** [Build Fast with AI](https://buildfastwithai.com/genai-course)",
        unsafe_allow_html=True
    )

# Display welcome message in chat format
with st.chat_message("assistant"):
    st.write("Ask me anything!")

# Initialize session ID if not exists
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Function to save message to database
def save_message_to_db(session_id, user_message, ai_message, thinking_process=None):
    if st.session_state.neon_database_url:
        try:
            engine = create_engine(st.session_state.neon_database_url)
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO chat_history (session_id, user_message, thinking_process, ai_message)
                    VALUES (:session_id, :user_message, :thinking_process, :ai_message)
                """), {
                    "session_id": session_id,
                    "user_message": user_message,
                    "thinking_process": thinking_process,
                    "ai_message": ai_message
                })
                conn.commit()
        except Exception as e:
            st.error(f"Failed to save to database: {str(e)}")

# Function to load chat history from database
def load_chat_history(session_id):
    if st.session_state.neon_database_url:
        try:
            engine = create_engine(st.session_state.neon_database_url)
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT user_message, thinking_process, ai_message 
                    FROM chat_history 
                    WHERE session_id = :session_id 
                    ORDER BY timestamp
                """), {"session_id": session_id})
                
                messages = [SystemMessage(content="You are a helpful AI assistant.")]
                for row in result:
                    user_message, thinking_process, ai_message = row
                    messages.append(HumanMessage(content=user_message))
                    
                    # Reconstruct AI message with thinking tags if thinking exists
                    full_response = ai_message
                    if thinking_process:
                        full_response = f"{thinking_process}\n{ai_message}"
                    messages.append(AIMessage(content=full_response))
                return messages
        except Exception as e:
            st.error(f"Failed to load chat history: {str(e)}")
    return None

# Initialize chat history if not exists
if "messages" not in st.session_state:
    try:
        # Make sure the connection string includes sslmode
        connection_string = f"{st.session_state.neon_database_url}?sslmode=require"
        
        history = PostgresChatMessageHistory(
            connection_string=connection_string,
            session_id=st.session_state.session_id,
            table_name="message_store"
        )
        
        # Load existing messages from Postgres
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant.")
        ] + history.messages
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant.")
        ]

# Display chat history
for message in st.session_state.messages[1:]:  # Skip the system message
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            content = message.content
            if "<think>" in content and "</think>" in content:
                # Extract thinking content
                think_start = content.index("<think>")
                think_end = content.index("</think>") + len("</think>")
                thinking = content[think_start:think_end]
                actual_response = content[think_end:].strip()
                
                # Display thinking in expandable section
                with st.expander("Show AI thinking process"):
                    st.write(thinking)
                
                # Display actual response
                st.write(actual_response)
            else:
                st.write(content)

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Check for API key in the input field directly
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
        thinking_placeholder = st.empty()  # Place thinking placeholder first
        message_placeholder = st.empty()   # Then message placeholder
        full_response = ""
        
        # Stream the response
        for chunk in chat.stream(st.session_state.messages):
            if chunk.content:
                full_response += chunk.content
                # Only update message placeholder during streaming if no thinking tags detected yet
                if "<think>" not in full_response:
                    message_placeholder.write(full_response)
        
        # After streaming is complete, check for thinking tags
        if "<think>" in full_response and "</think>" in full_response:
            # Extract thinking content
            think_start = full_response.index("<think>")
            think_end = full_response.index("</think>") + len("</think>")
            thinking = full_response[think_start:think_end]
            
            # Extract actual response
            actual_response = full_response[think_end:].strip()
            
            # First display thinking in expandable section
            with thinking_placeholder:
                with st.expander("Show AI thinking process"):
                    st.write(thinking)
            
            # Then display actual response
            message_placeholder.write(actual_response)
        else:
            # If no thinking tags, clear thinking placeholder and show full response
            thinking_placeholder.empty()
            message_placeholder.write(full_response)
    
    # Save messages to Postgres
    connection_string = f"{st.session_state.neon_database_url}?sslmode=require"
    history = PostgresChatMessageHistory(
        connection_string=connection_string,
        session_id=st.session_state.session_id,
        table_name="message_store"
    )
    history.add_user_message(prompt)
    history.add_ai_message(full_response)

    # Update session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=full_response))

    # After getting AI response
    thinking_process = None
    ai_message = full_response
    
    # Extract thinking content if present
    if "<think>" in full_response and "</think>" in full_response:
        think_start = full_response.index("<think>")
        think_end = full_response.index("</think>") + len("</think>")
        thinking_process = full_response[think_start:think_end]
        ai_message = full_response[think_end:].strip()
    
    # Save the complete conversation turn to database
    save_message_to_db(
        session_id=st.session_state.session_id,
        user_message=prompt,
        ai_message=ai_message,
        thinking_process=thinking_process
    )

