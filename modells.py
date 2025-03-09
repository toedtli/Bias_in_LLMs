import os
import openai
from openai import OpenAI
import google.generativeai as genai
import requests


class ModelAPI:
    def __init__(self):
        """Initialize API keys and configure models."""
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            raise ValueError("Missing Gemini API key. Set it as an environment variable.")
        genai.configure(api_key=self.GEMINI_API_KEY)

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("Missing OpenAI API key. Set it as an environment variable.")
        self.client = OpenAI(api_key=self.OPENAI_API_KEY) 


        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
        if not self.DASHSCOPE_API_KEY:
            raise ValueError("Missing Dashcope API key. Set it as an environment variable.")
        self.qwen_client = OpenAI(api_key=self.DASHSCOPE_API_KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1") 

        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        if not self.DEEPSEEK_API_KEY:
            raise ValueError("Missing Deepseek API key. Set it as an environment variable.")
        self.deepseek_client = OpenAI(api_key=self.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

        self.HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")


    def chat_with_gemini(self, prompt: str, system_message: str = None) -> str:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_message)
            response = model.generate_content(prompt)
            
            if hasattr(response, "text"):
                return response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "No response from Gemini"
        except Exception as e:
            return f"Gemini API Error: {e}"

    def chat_with_gpt(self, prompt: str, system_message: str = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT API Error: {e}"

    def chat_with_qwen(self, prompt: str, system_message: str = None) -> str:
        try:
            response = self.qwen_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": system_message or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Qwen API Error: {e}"
        
    def chat_with_deepseek(self, prompt: str, system_message: str = None) -> str:
        try:
            #You can invoke DeepSeek-V3 by specifying model='deepseek-chat'.
            #You can invoke DeepSeek-R1 by specifying model='deepseek-reasoner'.
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Deepseek API Error: {e}"
        
    def chat_with_falcon(self, prompt: str, system_message: str = None) -> str:
        try:
            api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'  # Replace with the desired model's endpoint
            api_token = self.HUGGING_FACE_API_KEY  # Replace with your Hugging Face API token

            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }

            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': 200,  # Adjust as needed
                    'return_full_text': False
                }
            }

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result[0].get('generated_text', 'No generated text found.')
            else:
                return f"Falcoin API request failed with status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Falcon error: {e}"
        
    def chat_with_bloom(self, prompt: str, system_message: str = None) -> str:
        try:
            api_url = 'https://api-inference.huggingface.co/models/bigscience/bloom'  # Replace with the desired model's endpoint
            api_token = self.HUGGING_FACE_API_KEY  # Replace with your Hugging Face API token

            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }

            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': 200,  # Adjust as needed
                    'return_full_text': False
                }
            }

            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                return result[0].get('generated_text', 'No generated text found.')
            else:
                return f"Bloom API request failed with status code {response.status_code}: {response.text}"
        except Exception as e:
            return f"Bloom error: {e}"
