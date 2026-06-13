import requests
import random

# fake MNIST-like input (784 numbers)
payload = {
    "input": [random.random() for _ in range(784)]
}

res = requests.post(
    "http://127.0.0.1:5000/predict",
    json=payload
)

print(res.json())
