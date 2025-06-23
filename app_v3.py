import streamlit as st
from google import genai
from google.genai import Client, types
import pandas as pd
import openpyxl
from st_aggrid import AgGrid
from google.cloud import aiplatform_v1beta1
import json
import os
import glob
from datetime import datetime, timedelta
import pdfplumber
from PIL import Image
import io
import tempfile
from dataclasses import asdict
from components.star_convo import star_conversation
from components.chat_repo import show_modal
from components.DigitalSalesReport import DigitalSalesReport
from components.prompts import prompt as pt 
from components.twolayer_sm import ReportConfig, ReportAnalyzer
from components.anomalies_detection_cy import DataAnomalyDetector 
from components.LinkDB import DataFromDatalab as db
from components.get_csv_metadata import get_csv_metadata
from components.date_bucket_agg import date_bucket_temp1, agg_report, linechart
from components.SendEmail import send_email
from components.readfiles import read_file

#pip install --proxy=fastweb.int.bell.ca xxxx
#cd "C:\Users\ca6107029\OneDrive - Bell Canada\2025\Automation\GoogleAI\Dev\chat-gen-bi-reports> "
#gcloud auth application-default login

# Simple hardcoded credentials (for demo only; use a secure method for production)
PASSWORD = "AIUSER"

def login():
    st.title("Agent BI Login")
    col1,col2 = st.columns([3,1])
    with col1: 
        username = st.text_input("Enter your Bell email")
    with col2:
        st.markdown("<div style='padding-top: 2.4em;'>@bell.ca</div>", unsafe_allow_html=True)

    if username : 
        email = f"{username}@bell.ca" #for use in the send email session
        st.session_state["email"] = email

    password = st.text_input("Password", value=PASSWORD, type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid credentials") 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- Chat Interface ---
st.set_page_config(layout="wide")  # Enables full-width display


# Initialize Gemini client
client = genai.Client(
    vertexai=True,
    project='prj-exp-sbx-apr-xqs7fz4g',
    location='us-central1'
)

# Create a "Restart Chats" button
if st.button("Restart Chats"):
    # Selectively reset chat-related session variables
    st.session_state.pop("chat_history", None)
    st.session_state.pop("current_df", None)    
    st.session_state.pop("current_report", None)

if "chat_repository" not in st.session_state:
    st.session_state["chat_repository"] = {}  # Initialize properly


# Sidebar for selecting reports
with st.sidebar:
    st.markdown("## Select BI Reports")  ## sm - two-layer report module

    analyzer = ReportAnalyzer()

    # Initialize session state variables for selections
    if "selected_report" not in st.session_state:
        st.session_state["selected_report"] = None
    if "selected_analyze_report" not in st.session_state:
        st.session_state["selected_analyze_report"] = None
    if "selected_self_serve" not in st.session_state:
        st.session_state["selected_self_serve"] = None
    if "current_report" not in st.session_state:  # Single reference variable
        st.session_state["current_report"] = None

    # First selection box
    selected_report = st.selectbox(
        "Choose a report:",
        ["Choose a report"] + list(analyzer.reports_config.keys()),
        help="Each report type has specialized executive and detailed analysis"
    )


    if selected_report != "Choose a report":
        st.session_state["selected_report"] = selected_report
        st.session_state["selected_analyze_report"] = None
        st.session_state["selected_self_serve"] = None
        st.session_state["current_report"] = selected_report  # Ensure main value reference

    # Second selection box
    st.markdown("## Analyze level of one data")
    selected_analyze_report = st.selectbox(
        "Choose a report:",
        ["Choose a report", "Digital Sales Daily Raw", "Registration Tokenized", "Digital Easy Index"]
    )

    if selected_analyze_report != "Choose a report":
        st.session_state["selected_analyze_report"] = selected_analyze_report
        st.session_state["selected_report"] = None
        st.session_state["selected_self_serve"] = None
        st.session_state["current_report"] = selected_analyze_report  # Main reference update

    # Third selection box (Button interaction)
    st.markdown("## Self - Serve Tool")
    if st.button("Update your data for analysis"):  # Placeholder button
        st.session_state["selected_self_serve"] = "DIY"
        st.session_state["selected_report"] = None
        st.session_state["selected_analyze_report"] = None
        st.session_state["current_report"] = "DIY"  # Final reference
    
    st.markdown("## Conversation Repository")

    # List saved conversations
    if st.session_state["chat_repository"]:
        selected_convo = st.selectbox("Select a conversation:", list(st.session_state["chat_repository"].keys()))
    else:
        selected_convo = None

    if selected_convo:
        show_modal(selected_convo, st.session_state["chat_repository"])

    # if selected_convo:
    #  with st.expander(f"Viewing: {selected_convo}"):
    #     st.write(st.session_state['chat_repository'][selected_convo])

    st.markdown("## Export Chat History")

    if st.button("Export Current Chat"):
        # Format chat messages for export
        chat_content = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state["chat_history"]])
        chat_html = """
        <html>
        <head>
        <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; }
        .user { color: #0066CC; margin-bottom: 10px; }
        .assistant { color: #008000; margin-bottom: 10px; }
        .message { border-bottom: 1px solid #ddd; padding: 6px; }
        </style>
        </head>
        <body>
        <h2>Chat History</h2>
        """

        for entry in st.session_state["chat_history"]:
            role = entry["role"].capitalize()
            css_class = "user" if role == "User" else "assistant"
            chat_html += f"""
            <div class="message {css_class}">
                <strong>{role}:</strong> {entry["content"]}
            </div>
            """

        chat_html += "</body></html>"

        # Store the conversation in session state
        convo_title = f"Conversation {len(st.session_state['chat_repository']) + 1}"
        st.session_state["chat_repository"][convo_title] = chat_content

        # Create a downloadable text file
        st.download_button(
            label="Download Chat History (HTML)",
            data=chat_html,
            file_name="chat_history.html",
            mime="text/html"
        )
    
    st.markdown("## Save your favourate convo")
    star_conversation()

    file_path = "data/starred_conversations.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            starred_conversations = json.load(f)

        if starred_conversations:
            # Create dropdown options with meaningful labels
            options = [
                f"{idx + 1}. {convo['username']} - {convo['session_time'][:16]}"
                for idx, convo in enumerate(starred_conversations)
            ]

            selected_idx = st.selectbox("⭐ Select a Starred Conversation:", options)

            # Get the corresponding conversation by index
            convo_idx = int(selected_idx.split(".")[0]) - 1
            convo = starred_conversations[convo_idx]
            st.markdown(f"**👤 User:** {convo['username']}")
            st.markdown(f"**🕒 Saved On:** {convo['session_time']}")
            st.markdown(f"**💬 User Said:**\n\n{convo['user']}")
            st.markdown(f"**🤖 Assistant Replied:**\n\n{convo['assistant']}")
        else:
                st.info("No starred conversations yet.")
    else:
        st.info("No starred conversations yet.")


# Create two columns (chat box 3:1 repository)
col1, spacer, col2 = st.columns([5, 0.5, 1]) 


with col2:

    st.markdown("### ⚙️ Features")

    st.markdown("""
    - 🔁 **2-Layer Reports**  
    Executive Summary + AI Chat  
    _Pre-trained with tab search, memory & context awareness_

    - 📈 **Level of One Analysis**  
    Quick link to Datalabs & Daily csvs  
    _auto reporting + auto-generated visuals_

        - 📊 **Smart Anomaly Detection**  
        Built-in models to highlight outliers  

        - 📧 **Email to Stakeholders**  
        One-click Outlook send

    - 🤖 **Self-Serve Chatbot**  
    Upload your file & talk to Gemini 2.0  
    _Multi-tab chat with file memory_

    - 💾 **Export Conversations**  
    _Download sessions_  
    View later via **Conversation Repository**

    - ⭐ **Starred Q&A**  
    Save insightful responses  
    _Send feedback directly to dev team_
    """)


#----------------------------------------------------------------------------------------------
#---------------------------------MAIN PAGE IN THE MIDDLE--------------------------------------
#----------------------------------------------------------------------------------------------
with col1: 
    st.title("Chat GEM2.0 - BI Reports")
    st.write(f"**Welcome {st.session_state['username']}! You can start questions on latest digital performance**")

#------------------two layer analysis ----------------------------------------
    file_list = None
    if st.session_state.get("current_report") == "Digital Sales Report":
        file_list = glob.glob("data/Digital Sales report*.xlsx")
    elif st.session_state.get("current_report") == "Digital Billboard Report":
        file_list = glob.glob("data/Digital_Billboard_Report.xlsx") 

    df = None
    if file_list:

        excel_file = max(file_list, key=os.path.getctime) 
        xls = pd.ExcelFile(excel_file)  
        default_sheet = xls.sheet_names[0] 

        sheet_names = xls.sheet_names
        selected_sheet = st.selectbox("Choose a sheet to load:", sheet_names)
        df = pd.read_excel(xls, sheet_name=selected_sheet)

        if selected_report != "Choose a report":
            report_config = analyzer.reports_config[selected_report]
        #--------part 1 two layer reporting -------------------------------------
        # Show report info - different info for different content
        st.subheader("📈 Report Info")
        st.write(f"**Description:** {report_config['description']}")
        
        st.subheader("🎯 Focus Areas")
        for area in report_config['focus_areas']:
            st.write(f"• {area}")
            
        # Download the selected report
        # excel_data = analyzer.create_excel_download(report_config) -- won't work with data subfolder
        if st.session_state.get("current_report") == "Digital Sales Report":
            with open("data/Digital Sales report.xlsx", "rb") as file:
                excel_data = file.read()
        elif st.session_state.get("current_report") == "Digital Billboard Report":
            with open("data/Digital_Billboard_Report.xlsx", "rb") as file:
                excel_data = file.read()

        st.download_button(
            label="📥 Download Original Excel Report",
            data=excel_data,
            file_name=f"{selected_report.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download original Excel file with all formatting preserved"
        )

        if st.button("🔄 Generate Executive Summary", use_container_width=True):
                with st.spinner("Generating executive summary..."):
                    data_context = df.to_string()
                    
                    exec_sentences = report_config["ai_instructions"]["executive_summary"]["default_sentences"]
                    summary = analyzer.generate_ai_content(
                        report_config, 
                        "executive_summary",
                        exec_sentences,
                        data_context
                    )
                    st.session_state.executive_summary = summary

        if hasattr(st.session_state, 'executive_summary'):
                st.write(st.session_state.executive_summary)

#--------part 2 interactive chat bot -------------------------------------
        st.subheader("🔍 Interactive Insights & Deep-dive")

        report = DigitalSalesReport()

        # Convert list of sheet names to a readable string
        # sheet_names_str = ", ".join(report.xls.sheet_names)
        # # Corrected markdown display
        # st.markdown(f"**Available sheets:** {sheet_names_str}")

        # Reset chat history when switching reports
        if "current_report" in st.session_state and st.session_state.get("previous_report") != st.session_state["current_report"]:
            st.session_state["chat_history"] = []  # Clear chat history
            st.session_state["previous_report"] = st.session_state["current_report"]  # Track last report
            st.session_state.executive_summary = None
            st.session_state.selected_report = None

        # Ensure chat history exists
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Initialize chat history with preset sentence
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "New weekly sales trend report is loaded. Now you can ask questions:"}
            ]

        # Display chat history **before** asking for user input
        for entry in st.session_state["chat_history"]:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

        # User input
        if prompt := st.chat_input("Type your question and press Enter..."):
            # Add user message to history
            st.session_state["chat_history"].append({"role": "user", "content": prompt})

            # Find the matching sheet
            sheet_name = report.find_matching_sheet(prompt)

            # Ensure a valid sheet is found
            if sheet_name and sheet_name in report.xls.sheet_names:
                df = report.load_dataframe(prompt)

                    # Convert ALL datetime-like columns to string format
                for col in df.select_dtypes(include=["datetime64", "object"]):
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)

                st.session_state["current_df"] = df  # Store latest valid DataFrame
            else:
                # Keep previous DataFrame if new prompt doesn't match
                df = st.session_state.get("current_df", None)

            if df is not None:
                file_contents = df.to_markdown(index=False)
            else:
                file_contents = "No matching sheet found."

                # Display the updated table view
            st.markdown("### Current Table View:")
            if df is not None:
                AgGrid(df)  # Always showing the latest valid df
            else:
                st.write("No matching sheet found.")


            with st.spinner("Fetching data..."):
                # Show key metrics
                key_metrics = report.get_metrics(prompt)
                st.write(f"📊 **Here are the key metrics included in the current table view for questioning: (Please wait while running)**")
                st.write(", ".join(key_metrics))


            # Compile chat history into the full prompt for model memory
            chat_history_str = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state["chat_history"]])

            # Compose prompt for Gemini (add data context if needed)
            full_prompt = f"Your main task: {pt('Maintask')}\n\nThis is the basic relationships among metrics: {pt('Sales_causes')}\n\nHere is the conversation history:\n{chat_history_str}\n\nUser's new question: {prompt}\n\n### Data Overview\n{file_contents}\n\n### Output format\n{pt('output_prompt')}"

            # Call Gemini model
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[full_prompt]
            )
            answer = response.text

            # Add Gemini response to history
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            # Display user message **immediately after submission**
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)


        # Function to count tokens
        if st.session_state["chat_history"]:  # Ensure chat history is not empty
            tokens = client.models.count_tokens(
                model="gemini-2.0-flash-lite",
                contents=[{"text": entry["content"]} for entry in st.session_state["chat_history"]]
            )
        else:
            tokens = 0  # Default to zero tokens if no chat history exists


        # Display token count below chat box
        st.info(f"You are using gemini-2.0-flash-001. Current token count in chat history: {tokens}")


#---------------------------------------------Update your data for analysis ----------------------------

    if st.session_state.get("current_report") == "DIY":
    # Load your data
        st.title("Upload Your Data")
        uploaded_file = st.file_uploader("Upload a CSV, Excel, or PDF file", type=["csv", "xlsx", "xls"])

        # Reset chat history when switching reports
        if "current_report" in st.session_state and st.session_state.get("previous_report") != st.session_state["current_report"]:
            st.session_state["chat_history"] = []  # Clear chat history
            st.session_state["previous_report"] = st.session_state["current_report"]  # Track last report

        # Ensure chat history exists
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Initialize chat history with preset sentence
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "Data is loaded. Now you can ask questions:"}
            ]


        if uploaded_file:
            
            # Ensure the 'uploads' directory exists before saving
            os.makedirs("uploads", exist_ok=True)

            upload_dir = "uploads"
            for item in glob.glob(os.path.join(upload_dir, "*")):
                try: 
                    if os.path.isfile(item):  # This skips directories like 'Samples'
                        os.remove(item)
                except PermissionError:
                    st.warning(f"⚠ File in use and couldn't be deleted: {item}")

            # Save the uploaded file temporarily # Store path in session
            temp_file_path = os.path.join("uploads", uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state["uploaded_file"] = temp_file_path  

            # Handle Excel files dynamically
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                # Find latest matching file in the directory
                #file_list = glob.glob("uploads/*.xlsx")  # Adjust pattern as needed
                excel_file = temp_file_path
                
                # Load the selected file
                with  pd.ExcelFile(excel_file, engine="openpyxl") as xls: 
                    sheet_names = xls.sheet_names
                    default_sheet = sheet_names[0]


                    if "selected_sheet" not in st.session_state:
                        st.session_state["selected_sheet"] = default_sheet

                    selected_sheet = st.selectbox("Select a sheet to preview:", sheet_names)
                    st.session_state["selected_sheet"] = selected_sheet

                    #df = pd.read_excel(excel_file, sheet_name=selected_sheet, engine="openpyxl")
                    df = pd.read_excel(xls, sheet_name=selected_sheet) ### this CAN WORK! for sas output - if error, try open and re-save as .xlsx , will rewrite the corrupted metadata fields into clean Office-compliant values

            else : df = read_file(uploaded_file)
        
            if df is not None:
                st.success("File loaded successfully!")
                # Optional: st.dataframe(df)  # to preview the content
            else:
                st.warning("Could not read the file. Make sure it's a supported format with tabular content.")


        if df is not None:
            st.subheader("Preview of Uploaded Data:")
            st.dataframe(df)
            file_contents = df.to_markdown(index=False)

            # Prompt example (displayed as markdown for formatting)
            st.markdown(f"""
            **Questions examples (preset as default):**

            Main task: You are an expert data and business intelligence analyst. 
            Use the data to analyze the week-over-week change in online sales for the most recent weeks, and potential drivers for the up or down trends.
            Output format: Keep executive level tone, be concise with key data points and actionable insights in bullet points. Maximum 5 lines."
            """)

            # Display chat history **before** asking for user input
            for entry in st.session_state["chat_history"]:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])

            # User input
            if prompt := st.chat_input("Type your question and press Enter...", key='DIY'):
                # Add user message to history
                st.session_state["chat_history"].append({"role": "user", "content": prompt})

                # Compile chat history into the full prompt for model memory
                chat_history_str = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state["chat_history"]])

                # Compose prompt for Gemini (add data context if needed)
                full_prompt = f"Your main task: {pt('Maintask')}\n\nThis is the basic relationships among metrics: {pt('Sales_causes')}\n\nHere is the conversation history:\n{chat_history_str}\n\nUser's new question: {prompt}\n\n### Data Overview\n{file_contents}\n\n### Output format\n{pt('output_prompt')}"

                # Call Gemini model
                response = client.models.generate_content(
                    model='gemini-2.0-flash-001',
                    contents=[full_prompt]
                )
                answer = response.text

                # Add Gemini response to history
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})

                # Display user message **immediately after submission**
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
            

            # Function to count tokens
            if st.session_state["chat_history"]:  # Ensure chat history is not empty
                tokens = client.models.count_tokens(
                    model="gemini-2.0-flash-lite",
                    contents=[{"text": entry["content"]} for entry in st.session_state["chat_history"]]
                )
            else:
                tokens = 0  # Default to zero tokens if no chat history exists


            # Display token count below chat box
            st.info(f"You are using gemini-2.0-flash-001. Current token count in chat history: {tokens}")

#--------------------------------------analysis level of one datasets--------------------------------------------------------
    if st.session_state.get("current_report") in ["Digital Sales Daily Raw", "Registration Tokenized", "Digital Easy Index"]:
        st.write("Current Report:", st.session_state.get("current_report"))

        # Reset chat history when switching reports
        if "current_report" in st.session_state and st.session_state.get("previous_report") != st.session_state["current_report"]:
            st.session_state["chat_history"] = []  # Clear chat history
            st.session_state["previous_report"] = st.session_state["current_report"]  # Track last report

        # Ensure chat history exists
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Initialize chat history with preset sentence
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "New weekly sales trend report is loaded. Now you can ask questions:"}
            ]
    
        db=db() #create an instance of the class

        if selected_analyze_report == "Digital Sales Daily Raw":
            df = pd.read_excel("data/Digital Sales Daily Raw.xlsx")
        elif selected_analyze_report == "Digital Easy Index":
            df = db.DEI_bell()
        elif selected_analyze_report == "Registration Tokenized":
            df = db.registration()

        # Close the connection after processing all cases
        db.close_connection()
            
        df, metadata_df, value_col_names = get_csv_metadata(df)

        if df is not None:
            # Display the DataFrame in Streamlit
            st.subheader(f"Preview of {selected_report}:")
            st.dataframe(df)
            st.subheader("Filter Data by Unique Values")

            if not value_col_names:
                st.warning("⚠️ No numeric columns available for selection.")
                st.stop()  # or return, if inside a function
            else:
                selected_value_col = st.selectbox("🎯 Select one value column", value_col_names)
                non_select_value_col = [col for col in value_col_names if col != selected_value_col]

            # **Dynamic filtering: exclude 'Column Name' & 'Data Type'**
            text_filter_columns = [col for col in metadata_df["Column Name"].dropna().unique().tolist() if col.strip() != "" and col in df.columns]

            # Dictionary to store selected filters
            selected_filters = {}

            # **Arrange filters in rows of 5**
            num_cols_per_row = 5
            columns = st.columns(num_cols_per_row)  # Create 5 columns for horizontal layout

            for i, col in enumerate(text_filter_columns):
                unique_values = df[col].dropna().unique().tolist()
                with columns[i % num_cols_per_row]:
                    selected_filters[col] = st.multiselect(f"{col}", options=unique_values)

            # Apply filters to the actual df
            filtered_df = df.copy()
            for col, values in selected_filters.items():
                if values:
                    filtered_df = filtered_df[filtered_df[col].isin(values)]

            # Rebuild list of valid columns that exist in the filtered_df
            all_valid_columns = filtered_df.columns.tolist()
            required_col = (["PERIOD_DT"] + [selected_value_col] + [col for col in text_filter_columns if col in all_valid_columns])
            filtered_df = filtered_df[required_col]

            # Get the current date
            filtered_df['PERIOD_DT'] = pd.to_datetime(filtered_df['PERIOD_DT'])
            current_date = filtered_df['PERIOD_DT'].max()

#---------------generate one row report based on selected value - frame into 6 weeks report format ----------------------
            today = datetime.now().date() 
            if today.weekday() == 0:  # Monday
                report_date = today - timedelta(days=2)  # Last Saturday
            else:
                report_date = today + timedelta(days=(5 - today.weekday()))  # Current Saturday

            st.write(f"📆 Latest date in filtered data: {current_date} Report data for latest Saturday: {report_date}" )
            st.dataframe(filtered_df.head())


            date_buckets = date_bucket_temp1 (report_date,current_date)
            agg_results = agg_report(date_buckets, filtered_df)

            agg_df = pd.DataFrame(agg_results)
            
            transposed_df = pd.DataFrame([agg_df.iloc[:, 1].values], columns=agg_df.iloc[:, 0].values)
            st.subheader("Aggregated Report Results")
            st.dataframe(transposed_df)  # Use st.write(agg_df) for a simpler view

#---------------apply models to the selected dataset ----------------------
            detector = DataAnomalyDetector(
                date_column='PERIOD_DT',
                category_columns=text_filter_columns,
                volume_columns=value_col_names
            )

            # UI layout with two buttons side by side
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🚨 Detect Anomalies"):
                    results = detector.analyze(filtered_df)
                    
                        # Flatten AnomalyResult objects into dicts
                    if isinstance(results, dict) and "all" in results:
                        flat_results = [asdict(a) for a in results["all"]]
                        results_df = pd.DataFrame(flat_results)
                    else:
                        st.error("Unexpected result format from anomaly detector.")
                        results_df = pd.DataFrame()  # fallback

                    st.success("✅ Anomaly detection complete!")
                    st.dataframe(results_df.head())  # Optional: Show a preview

                    # Optionally store results globally to reuse for email
                    st.session_state["latest_results"] = results_df

            with col2:
                if st.button("✉️ Send Email"):
                    # Pull results from session or regenerate if needed
                    results = st.session_state.get("latest_results")

                    if results is None:
                        st.warning("⚠️ Please detect anomalies first before sending the email.")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as temp_file:
                            results.to_csv(temp_file.name, index=False)
                            temp_file_path = temp_file.name

                        email = st.session_state["email"]
                        send_email(temp_file_path, email, "Anomaly Detection")
                        st.success("📬 Email sent successfully with attached anomaly report!")


#---------------create simple visualizations ----------------------
            linechart(agg_df)

        # Display chat history **before** asking for user input
        for entry in st.session_state["chat_history"]:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

        # User input
        if prompt := st.chat_input("Type your question and press Enter..."):
            # Add user message to history
            st.session_state["chat_history"].append({"role": "user", "content": prompt})


            if agg_df is not None:
                file_contents = agg_df.to_markdown(index=False)
            else:
                file_contents = "No matching sheet found."


            with st.spinner("Fetching data..."):
                # Show key metrics
                st.write(f"📊 **Please wait while the model is running**")

            # Compile chat history into the full prompt for model memory
            chat_history_str = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state["chat_history"]])

            # Compose prompt for Gemini (add data context if needed)
            full_prompt = f"Your main task: {pt('Maintask')}\n\nThis is the basic relationships among metrics: {pt('LOO_understand')}\n\nHere is the conversation history:\n{chat_history_str}\n\nUser's new question: {prompt}\n\n### Data Overview\n{file_contents}\n\n### Output format\n{pt('output_prompt')}"

            # Call Gemini model
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=[full_prompt]
            )
            answer = response.text

            # Add Gemini response to history
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            # Display user message **immediately after submission**
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(answer)


        # Function to count tokens
        if st.session_state["chat_history"]:  # Ensure chat history is not empty
            tokens = client.models.count_tokens(
                model="gemini-2.0-flash-lite",
                contents=[{"text": entry["content"]} for entry in st.session_state["chat_history"]]
            )
        else:
            tokens = 0  # Default to zero tokens if no chat history exists


        # Display token count below chat box
        st.info(f"You are using gemini-2.0-flash-001. Current token count in chat history: {tokens}")


