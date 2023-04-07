import face_recognition
import base64
from django.http import HttpRequest, JsonResponse, HttpResponse
import json
from PIL import Image
import numpy as np
import io

# an array of tuples of face encoder and labels
faces = []

def load_image(body):
    decoded_img = base64.b64decode(body)
    img_stream = io.BytesIO(decoded_img)
    img = Image.open(img_stream)
    img = np.asarray(img)
    return img



def predict(req: HttpRequest):
    if req.method == 'POST':
        body = req.body.decode('utf-8')
        body = json.loads(body)
        img = load_image(body['image'])
        unknown_face_encoding = face_recognition.face_encodings(img)[0]
        results = []
        for encoder, labels in faces:
            results = face_recognition.compare_faces([encoder], unknown_face_encoding)
            if(results[0]): 
                return JsonResponse({'label': labels}, safe=False)
        return JsonResponse({'label': None}, safe=False)

def register(req: HttpRequest):
    if req.method == 'POST':
        print('hi')
        body = req.body.decode('utf-8')
        body = json.loads(body)
        img = load_image(body['image'])
        faces.append([face_recognition.face_encodings(img)[0], body['label']])
        print('registered' + body['label'])
        return HttpResponse('success')

