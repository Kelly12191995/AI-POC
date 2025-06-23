import streamlit as st
import json
import os
from datetime import datetime

def star_conversation():
    # Ensure chat history exists
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    username = st.session_state["username"]

    # # Define floating button style
    # cartoon_button_style = """
    # <style>
    #     .floating-btn {
    #         position: fixed;
    #         left: 20px;
    #         bottom: 20px;
    #         background-color: #FFD700;
    #         color: black;
    #         font-family: 'Comic Sans MS', 'Chalkduster', cursive;
    #         font-size: 18px;
    #         padding: 10px 20px;
    #         border-radius: 50px;
    #         box-shadow: 4px 4px 6px rgba(0,0,0,0.3);
    #         border: none;
    #         cursor: pointer;
    #         z-index: 999;
    #     }
    #     .floating-btn:hover {
    #         background-color: #FFA500;
    #     }
    # </style>
    # """

    # # Inject styles
    # st.markdown(cartoon_button_style, unsafe_allow_html=True)

    # # Floating button (HTML to trigger hidden Streamlit button)
    # st.markdown("""
    #     <button class="floating-btn" onclick="document.getElementById('star_button').click()">⭐ Star Conversation</button>
    # """, unsafe_allow_html=True)

    # Capture timestamp
    session_time = datetime.now().strftime("%m/%d/%Y %I:%M%p")

# Save conversation when form is submitted
    if st.button("⭐ Star Conversation"):
        if len(st.session_state["chat_history"]) >= 2:
            last_user_message = st.session_state["chat_history"][-2]
            last_assistant_message = st.session_state["chat_history"][-1]

            starred_entry = {
                "username": username,  
                "session_time": session_time,  
                "user": last_user_message["content"],
                "assistant": last_assistant_message["content"]
            }

            file_path = "data/starred_conversations.json"

            # Load existing JSON file or create new list
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    starred_conversations = json.load(f)
            else:
                starred_conversations = []

            starred_conversations.append(starred_entry)

            # Save updated JSON
            with open(file_path, "w") as f:
                json.dump(starred_conversations, f, indent=4)

            st.success(f"Conversation starred successfully on {session_time}! ⭐")