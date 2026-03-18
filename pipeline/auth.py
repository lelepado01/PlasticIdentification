from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import json

SCOPES = ['https://www.googleapis.com/auth/drive']

flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
creds = flow.run_local_server(port=0)

# Save the token so you never need to do this again
with open('drive_token.json', 'w') as f:
    f.write(creds.to_json())

print("Authorisation complete. drive_token.json saved.")