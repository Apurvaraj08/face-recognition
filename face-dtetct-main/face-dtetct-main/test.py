from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.12.1:30233/v1"
)

completion = client.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    prompt="Where is Alaska?",
    max_tokens=50,
    temperature=0.7
)

print(completion)