import os
from google import genai

# Read the key from the environment — never hardcode credentials in source.
# PowerShell:  $env:GEMINI_API_KEY = "your-key"
# bash:        export GEMINI_API_KEY="your-key"
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise SystemExit("GEMINI_API_KEY is not set in the environment.")

try:
    # Modern official client from the Gemini ecosystem
    client = genai.Client(api_key=API_KEY)

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="Hello! If you can hear me, respond with: MODERN SDK IS WORKING"
    )
    print("\n🟢 SUCCESS:\n", response.text)

except Exception as e:
    print("\n🔴 API ERROR:", str(e))
