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

# Initialize embeddings and ChromaDB client
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
    return response.text.strip()

def check_and_summarize(summary: str) -> str:
    if count_tokens(summary) > TOKEN_THRESHOLD:
        return summarize_text(summary)
    return summary

# Inject custom CSS for a creative ChatGPT-like UI with icons
st.markdown(
    """
    <style>
    #chat_container {
      height: 500px;
      overflow-y: auto;
      border: 2px solid #ddd;
      padding: 15px;
      background: linear-gradient(135deg, #f0f9ff, #e0f7fa);
      border-radius: 8px;
    }
    .chat-message {
      margin: 10px 0;
      padding: 12px 15px;
      border-radius: 15px;
      max-width: 70%;
      word-wrap: break-word;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message.user {
      background-color: #DCF8C6;
      margin-left: auto;
      text-align: left;
      border: 1px solid #badbcc;
    }
    .chat-message.bot {
      background-color: #F1F0F0;
      margin-right: auto;
      text-align: left;
      border: 1px solid #ddd;
    }
    .chat-message.user::before {
      content: "👤 ";
      font-size: 1.2em;
    }
    .chat-message.bot::before {
      content: "🤖 ";
      font-size: 1.2em;
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

# Create a persistent container for chat history
chat_box = st.empty()

def display_chat(temp_message: str = ""):
    chat_html = '<div id="chat_container">'
    for speaker, message in st.session_state.chat_history:
        if speaker.lower() == "user":
            chat_html += f'<div class="chat-message user"><strong>User:</strong> {message}</div>'
        else:
            chat_html += f'<div class="chat-message bot"><strong>Bot:</strong> {message}</div>'
    # Append temporary streaming bot message if provided
    if temp_message:
        chat_html += f'<div class="chat-message bot"><strong>Bot:</strong> {temp_message}</div>'
    chat_html += "</div>"
    chat_box.markdown(chat_html, unsafe_allow_html=True)
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

# Display the persistent chat history container
display_chat()

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.conversation_summary = "No previous conversation."
    if Path(SUMMARY_FILE).exists():
        os.remove(SUMMARY_FILE)
    display_chat()

# Input area for user message
user_input = st.text_input("Your Message:", key="user_input_field")

if st.button("Send") and user_input:
    # Append the user's message
    st.session_state.chat_history.append(("User", user_input))
    display_chat()
    
    # Retrieve relevant context from ChromaDB
    results = query_chromadb(user_input)
    retrieved_docs = results.get("documents", [])
    flat_docs = [doc for sublist in retrieved_docs for doc in sublist]
    docs_str = "\n".join(flat_docs)
    
    # New prompt with reset instructions:
    prompt = (
        "You are an expert customer service assistant. Use the following information to answer the client's query.\n\n"
        f"Conversation History:\n{st.session_state.conversation_summary}\n\n"
        f"Client Query: {user_input}\n\n"
        "## Retrieved Context from Website Knowledge Base:\n"
        f"{docs_str}\n\n"
        "Provide a clear, accurate, and detailed answer. You can answer with an assumption if the retrieved context is incomplete or irrelevant to the user's query. "
        "When you need to assume details beyond the retrieved context, prepend your answer with the tag **[ASSUMPTION]** before mentioning any information not found in the context."
    )
    
    # Stream the bot's response while updating the same persistent chat container
    bot_response = ""
    stream = model.generate_content([prompt], stream=True)
    for chunk in stream:
        bot_response += chunk.text
        display_chat(temp_message=bot_response)
        time.sleep(0.1)
    
    # Append the final bot response and update the conversation summary
    st.session_state.chat_history.append(("Bot", bot_response))
    new_turn = f"User: {user_input}\nBot: {bot_response}"
    st.session_state.conversation_summary = update_conversation_summary(
        st.session_state.conversation_summary, [new_turn]
    )
    st.session_state.conversation_summary = check_and_summarize(st.session_state.conversation_summary)
    save_conversation_summary(st.session_state.conversation_summary)
    
    # Final update to the persistent chat container
    display_chat()
