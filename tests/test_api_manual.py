import requests

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "review": "This movie was amazing!"
    }
)

print(response.status_code)
print(response.json())