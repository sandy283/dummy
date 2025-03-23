import os
import time
from pathlib import Path
import streamlit as st
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

SUMMARY_FILE = "chat_summary.txt"
TOKEN_THRESHOLD = 2000

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

st.title("Customer Service Chatbot")

# Ask user to upload their Google Credentials JSON file
uploaded_credentials = st.file_uploader("Upload Google Credentials JSON", type="json")
if uploaded_credentials is not None:
    credentials_path = "google_credentials.json"
    with open(credentials_path, "wb") as f:
        f.write(uploaded_credentials.getbuffer())
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    # Initialize the generative model only after credentials are provided
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    st.error("Please upload your Google Credentials JSON file to continue.")
    st.stop()

def generate_answer_with_history(user_query: str, conversation_summary: str, delay: float = 0.1) -> str:
    results = query_chromadb(user_query)
    retrieved_docs = results.get("documents", [])
    flat_docs = [doc for sublist in retrieved_docs for doc in sublist]
    docs_str = "\n".join(flat_docs)
    
    prompt = (
        "You are an expert customer service assistant. Use the following information to answer the client's query.\n\n"
        f"Conversation History:\n{conversation_summary}\n\n"
        f"Client Query: {user_query}\n\n"
        "## Retrieved Context from Website Knowledge Base:\n"
        f"{docs_str}\n\n"
        "Provide a clear, accurate, and detailed answer. "
        "If the context is incomplete, mention any assumptions and suggest what additional details might be needed."
    )
    
    stream = model.generate_content([prompt], stream=True)
    answer_parts = []
    for chunk in stream:
        answer_parts.append(chunk.text)
        time.sleep(delay)
    return "".join(answer_parts)

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = load_conversation_summary()

user_input = st.text_input("Your Message:", key="user_input")

if st.button("Send") and user_input:
    st.session_state.chat_history.append(("User", user_input))
    
    placeholder = st.empty()
    bot_response = ""
    with placeholder.container():
        st.write("Bot: ")
        answer_chunks = []
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
        stream = model.generate_content([prompt], stream=True)
        for chunk in stream:
            answer_chunks.append(chunk.text)
            bot_response = "".join(answer_chunks)
            placeholder.text("Bot: " + bot_response)
            time.sleep(0.1)
    
    st.session_state.chat_history.append(("Bot", bot_response))
    
    new_turn = f"User: {user_input}\nBot: {bot_response}"
    st.session_state.conversation_summary = update_conversation_summary(st.session_state.conversation_summary, [new_turn])
    st.session_state.conversation_summary = check_and_summarize(st.session_state.conversation_summary)
    save_conversation_summary(st.session_state.conversation_summary)
    
    st.session_state.user_input = ""

for speaker, message in st.session_state.chat_history:
    if speaker == "User":
        st.markdown(f"**User:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
