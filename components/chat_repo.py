import streamlit as st

def show_modal(selected_convo, chat_repository):
    if "show_modal" not in st.session_state:
        st.session_state["show_modal"] = False

    if selected_convo:
        convo_text = chat_repository[selected_convo]

        modal_style = """
        <style>
        .modal {
            display: block;
            position: fixed;
            z-index: 1000;
            left: 30%;
            top: 20%;
            width: 40%;
            max-height: 400px;
            overflow-y: auto;
            background-color: white;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.2);
            padding: 20px;
            border-radius: 8px;
        }
        .modal-header {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }
        .close-button {
            cursor: pointer;
            font-size: 18px;
            color: red;
            background: none;
            border: none;
            font-weight: bold;
        }
        </style>
        """

        # Inject CSS
        st.markdown(modal_style, unsafe_allow_html=True)

        # Button to trigger modal
        if st.button("View Chat"):
            st.session_state["show_modal"] = True

        # Display modal if triggered
        if st.session_state["show_modal"]:
            st.markdown(f"""
            <div class="modal">
                <div class="modal-header">
                    Viewing: {selected_convo}
                    <form action="" method="post">
                        <button class="close-button" type="submit" name="close_modal">❌</button>
                    </form>
                </div>
                <div class="modal-body">
                    <p>{convo_text}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Close button (works within Streamlit)
            if st.button("Close Chat") or "close_modal" in st.session_state:
                st.session_state["show_modal"] = False
                st.rerun()  # Force immediate rerun to close chat
