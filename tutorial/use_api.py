import requests

url = 'http://127.0.0.1:5000/predict'

# Example data to test
data = {
    "features": [6.1, 2.8, 4.7, 1.2]
}

# Make a POST request
response = requests.post(url, json=data)

# Print the prediction result
print('Prediction:', response.json()['prediction'])