import os
import sys

# Add the directory containing modells.py to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from modells import ModelAPI

# Initialize the model API
modells = ModelAPI()

prompt = 'Hallo, alles klar?'
system_instruction = "Only answer in English. If the input is in another language, answer in English."

# Test Gemini
print(("-" * 40))
gemini_response = modells.chat_with_gemini(prompt, system_instruction)
print("\nGemini response: \n" + gemini_response)
print(("-" * 40))

# Test ChatGPT 
print(("-" * 40))
gpt_response = modells.chat_with_gpt(prompt, system_instruction)
print("GPT response: \n" + gpt_response)
print(("-" * 40))

# Test Qwen 
print(("-" * 40))
qwen_response = modells.chat_with_qwen(prompt, system_instruction)
print("Qwen response: \n" + qwen_response)
print(("-" * 40))

# Test Deepseek 
print(("-" * 40))
deepseek_response = modells.chat_with_deepseek(prompt, system_instruction)
print("Deepseek response: \n" + deepseek_response)
print(("-" * 40))






