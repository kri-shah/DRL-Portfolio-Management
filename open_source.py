import ollama
response = ollama.chat(
    model="qwen3",
    messages=[
        {"role": "user", "content": "How are you doing today? /think"}
    ]
)
print(response["message"]["content"])