import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import setting


def send_email_reset_pw(to_email, reset_link):
    body_message = (
        "You have requested a password reset. Please follow the URL below to create a new password:\n\n"
        f"{reset_link}\n\n"
        "If you did not request this, please ignore this email.\n\n"
        "Best regards\n\n"
        "MIND - Motions insights and diagnostics\n"
        "maxon motor ag\n"
        "www.maxongroup.com"
    )

    msg = MIMEMultipart()
    msg['From'] = setting.smtp_from
    msg['To'] = to_email
    msg['Subject'] = "Password Reset"
    msg.attach(MIMEText(body_message, 'plain'))

    with smtplib.SMTP(setting.smtp_server, setting.smtp_port) as server:
        server.starttls()
        server.send_message(msg)
