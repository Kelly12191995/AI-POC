# Agent BI Streamlit App

This is an interactive analytics and chat application for BCE BI reports, built with [Streamlit](https://streamlit.io/).  
It provides secure access, interactive chat with Gemini AI, and dynamic data exploration for digital sales and care flows.

---

## 📁 Folder Structure

```
chat-gen-bi-reports/
├── app_v3.py                   # Main Streamlit app
├── README.md
├── requirements.txt
├── debug.ipynb                 # Jupyter notebook for debugging and experiments
├── collab with git.docx
├── GitLab Setup Guide for VS Code.docx
├── assets/
├── components/                 # Custom Python modules for app features
│   ├── __init__.py
│   ├── anomalies_detection_cy.py   # Data anomaly detection
│   ├── chat_repo.py                # Conversation repository modal
│   ├── date_bucket_agg.py          # Date bucketing and aggregation
│   ├── DigitalSalesReport.py       # Digital sales report logic
│   ├── get_csv_metadata.py         # CSV metadata extraction
│   ├── LinkDB.py                   # Data loading functions (e.g., DEI_care)
│   ├── my_input_for_analysis.py    # Example for anomaly detection input
│   ├── prompts.py                  # Prompt templates for AI
│   ├── readfiles.py                # File reading (CSV, Excel, PDF)
│   ├── SendEmail.py                # Email sending utilities
│   ├── star_convo.py               # Starred conversation logic
│   ├── twolayer_sm.py              # Two-layer report config and analysis
│   └── __pycache__/
├── data/
│   ├── __init__.py
│   ├── Digital Sales Daily Raw.xlsx
│   └── starred_conversations.json  # Saved starred conversations
├── logs/
│   └── app_v2_history1.txt
├── pages/
├── uploads/                      # Temporary uploaded files for DIY analysis
└── __pycache__/
```

---

## 🚀 How to Run

1. **Install dependencies**  
   Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```

2. **Start the Streamlit app**  
   ```
   streamlit run app_v3.py
   ```

3. **Login**  
   Use the credentials set in `app_v3.py` (default: `AIUSER`).

4. **Explore**  
   - Use the sidebar to select reports or upload data.
   - Interact with the chat box to ask questions about your data.
   - View, star, and export chat history.

---

## 🗂️ Data

- Place your original Excel/CSV files in `data/`.
- Starred conversations are saved in `data/starred_conversations.json`.
- Uploaded files for DIY analysis are stored in `uploads/`.

---

## 🛠️ Features

- **Authentication:** Simple login screen for access control.
- **Interactive Chat:** Chat with Gemini AI, with scrollable history and token count.
- **Data Display:** Embedded, interactive Excel/CSV tables using AgGrid.
- **Prompt Examples:** Suggested questions for best results.
- **Conversation Repository:** Save, select, and export chat histories.
- **Starred Conversations:** Mark and review important conversations.
- **Modular Reports:** Switch between different report types via sidebar.
- **DIY Analysis:** Upload your own data for ad hoc analysis.
- **Anomaly Detection:** Detects anomalies in your data.
- **Email Integration:** Send reports or results via email.
- **PDF Table Extraction:** Extract tables from PDF files using `pdfplumber`.

---

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for full list

---

## 📄 Notes

- For production, replace hardcoded credentials with a secure authentication method.
- Ensure your Google Cloud credentials are set up for Gemini/Vertex AI access.
- Add your own modules to `components/` as needed.

---

*For questions or support, contact the project maintainer.*


1) Two layer BI reporting insights (*Executive Summary* + *AI chat*: pre-trained with ability to search tabs/business understanding/memory)
2) Level of one reporting (*Link to Datalabs* - ability to create simple trend report from daily files + visualization)
3) Self - Serve Chat bot (allow upload your own file and interact with Gemini 2.0 with chat memory and multiple tabs)
4) Export chat session (ability to *export/download* the file - and view it later in *Conversation Repository*)
5) Star Conversation (able to save your favourite AI response and questions - feedbacks to develop team)
6) Modelling enhancement (model to detect anomalies from level of one datasets )
7) Email link to outlook (able to send emails to stakeholders)