import os
import time
from pathlib import Path
import streamlit as st
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Constants
SUMMARY_FILE = "chat_summary.txt"
TOKEN_THRESHOLD = 2000

# Initialize embedding and ChromaDB client
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="chromadb_data"
))
collection = client.get_or_create_collection("processed_docs")

def query_chromadb(user_query: str):
    query_embedding = embedder.encode(user_query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results

def update_conversation_summary(current_summary: str, conversation_buffer: list) -> str:
    new_turns = "\n".join(conversation_buffer)
    if current_summary.strip() == "No previous conversation.":
        return new_turns
    else:
        return current_summary + "\n" + new_turns

def load_conversation_summary() -> str:
    if Path(SUMMARY_FILE).exists():
        with open(SUMMARY_FILE, "r") as f:
            return f.read()
    return "No previous conversation."

def save_conversation_summary(summary: str):
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)

def count_tokens(text: str) -> int:
    return len(text.split())

def summarize_text(text: str) -> str:
    prompt = (
        "You are an assistant that summarizes conversations concisely. "
        "Summarize the conversation below into a clear and concise summary while retaining key details.\n\n"
        f"Conversation:\n{text}\n\n"
        "Summary:"
    )
    response = model.generate_content([prompt])
    summarized = response.text.strip()
    return summarized

def check_and_summarize(summary: str) -> str:
    if count_tokens(summary) > TOKEN_THRESHOLD:
        st.info("Conversation summary exceeds token threshold. Summarizing...")
        return summarize_text(summary)
    return summary

# Custom CSS for ChatGPT-like appearance
st.markdown(
    """
    <style>
    #chat_container {
      height: 500px;
      overflow-y: auto;
      border: 1px solid #ccc;
      padding: 10px;
      background-color: #f9f9f9;
    }
    .chat-message {
      margin: 10px 0;
      padding: 10px;
      border-radius: 10px;
      max-width: 70%;
      word-wrap: break-word;
    }
    .chat-message.user {
      background-color: #DCF8C6;
      margin-left: auto;
      text-align: right;
    }
    .chat-message.bot {
      background-color: #F1F0F0;
      margin-right: auto;
      text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Customer Service Chatbot")

# Upload Google Credentials JSON file
uploaded_credentials = st.file_uploader("Upload Google Credentials JSON", type="json")
if uploaded_credentials is not None:
    credentials_path = "google_credentials.json"
    with open(credentials_path, "wb") as f:
        f.write(uploaded_credentials.getbuffer())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    st.error("Please upload your Google Credentials JSON file to continue.")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = load_conversation_summary()

# Create two columns: left for chat history, right for input
left_col, right_col = st.columns([3, 1])
left_container = left_col.empty()  # Container for the single chat box

def display_chat(temp_message: str = None):
    chat_html = '<div id="chat_container">'
    for speaker, message in st.session_state.chat_history:
        if speaker.lower() == "user":
            chat_html += f'<div class="chat-message user"><strong>User:</strong> {message}</div>'
        else:
            chat_html += f'<div class="chat-message bot"><strong>Bot:</strong> {message}</div>'
    # Append temporary bot message (streaming) if provided
    if temp_message:
        chat_html += f'<div class="chat-message bot"><strong>Bot:</strong> {temp_message}</div>'
    chat_html += "</div>"
    left_container.markdown(chat_html, unsafe_allow_html=True)
    st.components.v1.html(
        """
        <script>
          var chatContainer = document.getElementById('chat_container');
          if(chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
          }
        </script>
        """,
        height=0,
    )

with right_col:
    user_input = st.text_input("Your Message:", key="user_input_field")
    if st.button("Send") and user_input:
        # Append user's message and update chat history display
        st.session_state.chat_history.append(("User", user_input))
        display_chat()

        # Retrieve context from ChromaDB
        results = query_chromadb(user_input)
        retrieved_docs = results.get("documents", [])
        flat_docs = [doc for sublist in retrieved_docs for doc in sublist]
        docs_str = "\n".join(flat_docs)
        
        prompt = (
            "You are an expert customer service assistant. Use the following information to answer the client's query.\n\n"
            f"Conversation History:\n{st.session_state.conversation_summary}\n\n"
            f"Client Query: {user_input}\n\n"
            "## Retrieved Context from Website Knowledge Base:\n"
            f"{docs_str}\n\n"
            "Provide a clear, accurate, and detailed answer. "
            "If the context is incomplete, mention any assumptions and suggest what additional details might be needed."
        )
        
        # Stream bot response while updating the single chat container
        bot_msg = ""
        stream = model.generate_content([prompt], stream=True)
        for chunk in stream:
            bot_msg += chunk.text
            display_chat(temp_message=bot_msg)
            time.sleep(0.1)
        
        # Append bot response to conversation history
        st.session_state.chat_history.append(("Bot", bot_msg))
        new_turn = f"User: {user_input}\nBot: {bot_msg}"
        st.session_state.conversation_summary = update_conversation_summary(
            st.session_state.conversation_summary, [new_turn]
        )
        st.session_state.conversation_summary = check_and_summarize(
            st.session_state.conversation_summary
        )
        save_conversation_summary(st.session_state.conversation_summary)
        # Optionally, clear input by using a different key or instructing the user to overwrite the field.
