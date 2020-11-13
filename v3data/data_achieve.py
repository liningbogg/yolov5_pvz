import requests
response = requests.get('http://192.168.0.104/ocr/getYolov3Data/?username=DDG&shape_size=640')
print(response.text)
