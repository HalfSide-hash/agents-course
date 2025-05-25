import os
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")

# As seen in the LLM section, if we just do decoding, **the model will only stop when it predicts an EOS token**, 
# and this does not happen here because this is a conversational (chat) model and we didn't apply the chat template it expects.

output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Time to wait for US passport is..."},
    ],
    stream=False,
    max_tokens=1024,
)
print(output.choices[0].message.content)