import pandas as pd
import pyodbc
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from datetime import date
from datetime import datetime
import win32com.client as win32
from pathlib import Path
import pythoncom

def send_email(file_path, recipient_email, filename):
    pythoncom.CoInitialize()  # Initialize COM for this thread

    """
    Sends an email with the given file attached.
    
    Args:
        file_path (str): Path to the file to be attached.
        
    Notes:
        - Uses predefined recipient email list.
        - Formats the email in HTML with a standard message.
        - Sends via Outlook.
    """
    
    #file_name = os.path.basename(file_path)
    file_name = filename

    # Define recipients and sender
    recipient_list = [recipient_email]
    sender_email ="aidigitaladvocacyplatform@bell.ca"
    
    today_date = date.today()
    subject = f"{file_name} - {today_date}"

    # Initialize Outlook email
    ol = win32.Dispatch("outlook.application")
    newmail = ol.CreateItem(0)  # 0 = Mail item

    newmail.Subject = subject
    newmail.To = "; ".join(recipient_list)  # Send to multiple recipients
    newmail.SendUsingAccount = [acc for acc in ol.Session.Accounts if acc.SmtpAddress == sender_email][0]



    # Create HTML email body
    newmail.HTMLBody = f"""
        <html>
        <head></head>
        <body>
            <p>Hi All,</p>
            
            <p>Please find the attached data report.</p>

            <p>Run date: {today_date}</p>

            <p>If you would like to subscribe or unsubscribe to this report, please reply to the email to be added. </p>

            <p>Regards,</p>

            <p>
                <b>AI Digital Advocacy Platform</b><br>
                Email: <a href='mailto:carol.li@bell.ca'>carol.li@bell.ca</a><br>
                <b>Mission Statement:</b> Centralize | Learn faster | Best practices | Alignment | Innovation<br>
                One version of the truth & Shared vision for the future<br>
                <span style="color:red;">Confidential - For Internal Use Only</span>
            </p>
        </body>
        </html>
        """

    # Attach the file
    newmail.Attachments.Add(file_path)

    # Send the email
    newmail.Send()
    print(f"Email with attachment ({file_path}) sent to {', '.join(recipient_list)}.")

# Example Usage:
# send_email(r"C:\Users\ca6107029\OneDrive - Bell Canada\2025\Automation\GoogleAI\Agent BI\Agent BI\streamlit_app\data\Digital_Billboard_Report.xlsx")