import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, BackgroundTasks, HTTPException
import os

# ----------------------------------------------------------------------
# INSTRUCTIONS:
# 1. Install dependencies if needed: pip install fastapi pydantic uvicorn
# 2. Add this code into your existing Python Backend running on Azure.
# 3. IMPORTANT: You MUST generate a "Google App Password" for your Cassandra Gmail:
#    - Go to Google Account Settings -> Security -> 2-Step Verification
#    - Scroll down to "App Passwords"
#    - Create a new App Password (e.g. name it "Cassandra Web API")
#    - Paste that 16-character password into GMAIL_APP_PASSWORD below.
# ----------------------------------------------------------------------

# NOTE: In production, store these in Environment Variables (e.g., .env file)
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "cassandra.classcompanion@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "flku ebgy foac viiy")

# Define the expected JSON payload format from the frontend
class EmailNotificationPayload(BaseModel):
    emails: List[str]
    subject: str
    html_content: str
    course_name: str
    teacher_name: str

# Use the existing FastAPI 'app' variable if you have one, 
# otherwise this initializes it (e.g., app = FastAPI())
# app = FastAPI() 

def send_email_sync(payload: EmailNotificationPayload):
    """
    Synchronous function that handles the SMTP connection.
    Runs in a background thread to prevent API lag.
    """
    if not payload.emails:
        print("No emails provided to send_email_sync")
        return

    try:
        # 1. Prepare the email message envelope
        msg = MIMEMultipart("alternative")
        msg["Subject"] = payload.subject
        msg["From"] = f"Cassandra AI <{GMAIL_ADDRESS}>"
        
        # NOTE: We use BCC (Blind Carbon Copy) so students cannot see 
        # each other's email addresses! Highly important for privacy.
        msg["Bcc"] = ", ".join(payload.emails)

        # 2. Build the styled HTML email body
        styled_content = f"""
        <html>
            <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f7f9; padding: 20px; color: #0f172a;">
                <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <!-- Header -->
                    <div style="background: linear-gradient(135deg, #0060e7 0%, #004ba8 100%); padding: 25px; text-align: center; color: white;">
                        <h2 style="margin: 0; font-size: 24px;">{payload.course_name}</h2>
                        <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;">Instructor: {payload.teacher_name}</p>
                    </div>
                    
                    <!-- Content Body -->
                    <div style="padding: 30px; font-size: 16px; line-height: 1.6; border-bottom: 1px solid #e2e8f0;">
                        {payload.html_content}
                    </div>
                    
                    <!-- Footer -->
                    <div style="background-color: #F8FAFC; padding: 15px; text-align: center;">
                        <p style="margin: 0; color: #64748b; font-size: 12px;">
                            You are receiving this notification because you are enrolled in <strong>{payload.course_name}</strong>.
                        </p>
                        <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 11px;">
                            Powered by Cassandra AI Class Companion.
                        </p>
                    </div>
                </div>
            </body>
        </html>
        """

        # Attach the HTML body to the email envelope
        msg.attach(MIMEText(styled_content, "html"))

        # 3. Connect to Gmail's SMTP Server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            # Uncomment for detailed connection logs during debugging:
            # server.set_debuglevel(1)
            
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.send_message(msg)
            
        print(f"Successfully sent notification emails to {len(payload.emails)} students.")

    except Exception as e:
        print(f"Failed to send email via SMTP. Error: {str(e)}")

# Add this route to your existing Azure Python Backend API
@app.post("/api/send-notification")
async def api_send_notification(payload: EmailNotificationPayload, background_tasks: BackgroundTasks):
    """
    Endpoint triggered by the frontend whenever an assignment or announcement is made.
    """
    # Verify we have credentials before attempting
    if GMAIL_ADDRESS == "your_cassandra_email@gmail.com":
        raise HTTPException(status_code=500, detail="Server SMTP credentials are not configured.")

    # We use BackgroundTasks so the frontend gets an immediate "Success" response,
    # and doesn't have to wait 2-3 seconds for Google's SMTP servers to finish sending.
    background_tasks.add_task(send_email_sync, payload)
    
    return {
        "status": "success", 
        "message": f"Queued emails for {len(payload.emails)} students"
    }