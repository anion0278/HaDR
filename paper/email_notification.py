import win32com.client as win32
import time

def send_email(subject, text, address):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = address
    mail.Subject = subject
    mail.Body = text
    mail.HTMLBody = f'<h2>{text}</h2>'
    mail.Send()
    time.sleep(1)
    print("Sent email to:" + address)
    print("Email content:" + text)
    # mapi = outlook.GetNameSpace("MAPI")
    # inbox = mapi.GetDefaultFolder(4)
    # inbox.Folders.Session.SendAndReceive(True)
    # time.sleep(10)

