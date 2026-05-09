import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

try:
    print(f"Testing generation with model: {model_name}")
    response = model.generate_content("Hello, how are you?")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error during generation: {e}")
