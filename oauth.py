import os.path

import gensim
import json
from google.auth.transport.requests import Request
from google.oauth2._credentials_async import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
SECRET_FILE = os.path.join(test_data_dir, "client_secret.json")

# ran into error with oauth login - using token file to avoid separate login (this is for local server)
enable_login = os.path.join(test_data_dir, 'token.json')


def get_g_service(service="gmail", ver="v1",
                  scopes=['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send'
                          ]):
    creds = None
    if os.path.exists(enable_login):
        creds = Credentials.from_authorized_user_file(enable_login, scopes)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(SECRET_FILE, scopes)
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
    with open(enable_login, 'w') as token:
        token.write(creds.to_json())

    return build(service, ver, credentials=creds)
