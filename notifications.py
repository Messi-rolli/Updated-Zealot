# Email and SMS notifications
import os
import smtplib
from email.message import EmailMessage
from twilio.rest import Client

def send_email(subject, body, to):
    user = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASSWORD')
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    msg['from'] = user
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()

def send_sms(body, from_, to):
    account_ssid = os.environ.get('TWILIO_SSID')
    auth_token = os.environ.get('TWILIO_TOKEN')
    client = Client(account_ssid, auth_token)
    client.messages.create(body=body, from_=from_, to=to)
