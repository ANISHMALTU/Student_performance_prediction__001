from litellm import completion

response = completion(
    model="openrouter/openai/gpt-4.1-mini",
    messages=[{"role": "user", "content": "Say hello"}]
)

print(response["choices"][0]["message"]["content"])
