import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Height':20, 'VB':6, 'Tilt':5})

print(r.json())