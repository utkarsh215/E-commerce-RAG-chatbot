import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import pymongo
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Voice assistance imports
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import io

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="E-commerce Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fixed bottom chat input and voice integration
st.markdown("""
<style>
/* Fixed bottom container styling */
.fixed-chat-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-top: 1px solid #e0e0e0;
    padding: 15px 20px;
    z-index: 999;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Chat input styling */
.chat-input-container {
    display: flex;
    align-items: center;
    max-width: 800px;
    margin: 0 auto;
    background: #f8f9fa;
    border-radius: 25px;
    padding: 8px 15px;
    border: 2px solid #e9ecef;
    transition: border-color 0.2s ease;
}

.chat-input-container:focus-within {
    border-color: #007bff;
}

/* Voice button styling */
.voice-button {
    background: none;
    border: none;
    padding: 8px;
    margin-right: 10px;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.voice-button:hover {
    background: #e9ecef;
}

.voice-button.recording {
    background: #ff4444;
    color: white;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

/* Add bottom margin to main content to avoid overlap */
.main .block-container {
    padding-bottom: 120px;
}

/* Ensure chat messages are scrollable */
div[data-testid="stVerticalBlock"] {
    max-height: calc(100vh - 200px);
    overflow-y: auto;
}

/* Style for better chat appearance */
div[data-testid="chatMessage"] {
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Database configuration
@st.cache_resource
def init_database():
    DB_CONNECT = os.getenv("CONNECTION_DB")
    GEMINI_API = os.getenv("GEMINI_API")
    os.environ["GOOGLE_API_KEY"] = GEMINI_API
    
    client = pymongo.MongoClient(DB_CONNECT)
    db = client.E_Commerce
    collection = db.Products
    chat_history_collection = db.chat_history
    
    return client, db, collection, chat_history_collection

client, db, collection, chat_history_collection = init_database()

# Initialize AI components
@st.cache_resource
def init_ai_components():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0, max_tokens=790)
    
    vectorStore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name="vector_index",
        text_key="embedding_text",
        embedding_key="embedding"
    )
    
    retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    system_prompt = (
        "You are a helpful assistant that provides product recommendations and information. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum.\n\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return llm, retriever, qa_chain

llm, retriever, qa_chain = init_ai_components()

# Voice assistance functions
def text_to_speech(text):
    """Convert text to speech using gTTS and return audio data"""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Helper functions
def save_message_to_db(session_id, message, sender="user", response=None):
    """Save message to MongoDB chat history"""
    chat_doc = {
        "session_id": session_id,
        "message": message,
        "sender": sender,
        "timestamp": datetime.utcnow(),
        "response": response
    }
    chat_history_collection.insert_one(chat_doc)

def get_chat_history(session_id, limit=50):
    """Retrieve chat history for a session"""
    history = list(chat_history_collection.find(
        {"session_id": session_id}
    ).sort("timestamp", 1).limit(limit))
    
    return [{
        "message": doc["message"],
        "sender": doc["sender"],
        "timestamp": doc["timestamp"],
        "response": doc.get("response")
    } for doc in history]

def get_all_sessions():
    """Get all unique sessions with their latest message"""
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$session_id",
                "latest_message": {"$first": "$message"},
                "latest_timestamp": {"$first": "$timestamp"},
                "message_count": {"$sum": 1}
            }
        },
        {"$sort": {"latest_timestamp": -1}},
        {"$limit": 20}
    ]
    
    sessions = list(chat_history_collection.aggregate(pipeline))
    return sessions

# Process chat message
def process_message(prompt):
    """Process user message and generate response"""
    if not prompt.strip():
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Save user message to database
    save_message_to_db(st.session_state.session_id, prompt, "user")
    
    try:
        # Create memory and load previous conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load previous chat history into memory
        history = get_chat_history(st.session_state.session_id, limit=10)
        for item in history[:-1]:  # Exclude the current message we just saved
            if item["sender"] == "user" and item.get("response"):
                memory.chat_memory.add_user_message(item["message"])
                memory.chat_memory.add_ai_message(item["response"])
        
        # Create conversational RAG chain with memory
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False
        )
        
        # Get response from the RAG chain
        result = rag_chain.invoke({"question": prompt})
        
        # Extract answer from result
        if isinstance(result, dict) and "answer" in result:
            response = result["answer"]
        elif hasattr(result, "answer"):
            response = result.answer
        else:
            response = str(result)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update the user message document with the bot response
        chat_history_collection.update_one(
            {"session_id": st.session_state.session_id, "message": prompt, "sender": "user"},
            {"$set": {"response": response}},
            upsert=False
        )
        
    except Exception as e:
        error_message = f"I encountered an error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# Sidebar for session management
with st.sidebar:
    st.title("🛒 Product Chat")
    
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.subheader("Previous Chats")
    sessions = get_all_sessions()
    
    if sessions:
        for session in sessions[:10]:  # Show last 10 sessions
            session_id = session["_id"]
            latest_message = session["latest_message"]
            
            # Truncate message for display
            display_message = latest_message[:40] + "..." if len(latest_message) > 40 else latest_message
            
            if st.button(f"{display_message}", key=session_id, use_container_width=True):
                # Load selected session
                st.session_state.session_id = session_id
                history = get_chat_history(session_id)
                
                # Rebuild messages for display
                st.session_state.messages = []
                for item in history:
                    if item["sender"] == "user":
                        st.session_state.messages.append({"role": "user", "content": item["message"]})
                    if item.get("response"):
                        st.session_state.messages.append({"role": "assistant", "content": item["response"]})
                st.rerun()
    else:
        st.info("No previous chats found")

# Main chat interface
st.title("🛒 E-commerce Assistant")
st.markdown("Ask me about products, get recommendations, or explore product details!")

# Chat messages display
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add text-to-speech button for assistant messages
        if message["role"] == "assistant":
            if st.button("🔊", key=f"tts_{i}", help="Listen to this response"):
                audio_data = text_to_speech(message["content"])
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")

# Enhanced voice-integrated chat input using st._bottom container
with st._bottom:
    st.markdown('<div class="fixed-chat-container">', unsafe_allow_html=True)
    
    # Create columns for voice and text input
    col1, col2 = st.columns([1, 6])
    
    with col1:
        # Voice input with auto-submit
        voice_text = speech_to_text(
            language="en",
            start_prompt="🎤",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=True,
            key="voice_input_integrated"
        )
    
    with col2:
        # Regular chat input
        prompt = st.chat_input("Type your message or click the microphone...")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle voice input with auto-submit
    if voice_text:
        process_message(voice_text)
        st.rerun()
    
    # Handle text input
    if prompt:
        process_message(prompt)
        st.rerun()

# Footer
# st.markdown("---")
# col1, col2 = st.columns(2)
# with col1:
#     st.markdown("**Current Session ID:** `" + st.session_state.session_id + "`")
# with col2:
#     st.markdown("")