import ollama

stream = ollama.generate(
    model="mistral", 
    prompt="hey how are you",
    raw=True,
)

response1 = ""
for chunk in stream:
    response1 += chunk['response']
    print(chunk['response'], end='', flush=True)
