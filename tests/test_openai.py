"""
Test script to verify Azure OpenAI configuration and connectivity.
Run this to debug your .env settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 50)
print("Azure OpenAI Configuration Test")
print("=" * 50)

# 1. Check environment variables
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("DEPLOYMENT_NAME")

print(f"\nEndpoint: {endpoint}")
print(f"Deployment Name: {deployment}")
print(f"API Key: {'***' + api_key[-4:] if api_key else 'NOT SET'}")

if not all([endpoint, api_key, deployment]):
    print("\n❌ ERROR: Missing environment variables!")
    exit(1)

print("\n✅ All environment variables are set.")

# 2. Test connection
print("\n" + "=" * 50)
print("Testing Azure OpenAI Connection...")
print("=" * 50)

try:
    from openai import OpenAI
    
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )
    
    print(f"\nSending test request to deployment: '{deployment}'...")
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "user", "content": "Say 'Hello, connection successful!' in exactly those words."}
        ],
        max_tokens=50
    )
    
    print(f"\n✅ SUCCESS! Response received:")
    print(f"   {response.choices[0].message.content}")
    print(f"\n   Model used: {response.model}")
    print(f"   Tokens used: {response.usage.total_tokens}")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   {e}")
    print("\n" + "-" * 50)
    print("Troubleshooting tips:")
    print("1. Check if DEPLOYMENT_NAME matches a GPT deployment in Azure Portal")
    print("2. Ensure the deployment is a chat model (gpt-4, gpt-4o, gpt-35-turbo)")
    print("3. Verify the API key is correct")
    print("4. Check if the endpoint URL is correct")
    print("-" * 50)
