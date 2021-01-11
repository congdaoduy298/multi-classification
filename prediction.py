from PIL import Image 
from io import BytesIO
from tensorflow.keras.models import model_from_json
from face_recognition import face_locations
import numpy as np 
import time 
import pandas as pd 

# INPUT_SHAPE_1 = (480, 480)
INPUT_SHAPE_2 = (224, 224)

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def detect_face(image: Image.Image):
    # image = image.resize(INPUT_SHAPE_1)
    image = image.resize(INPUT_SHAPE_2)
    arr_image = np.asarray(image)
    s = time.time()
    faces = face_locations(arr_image, model='cnn')
    e = time.time()
    print(f'Detection time : {e-s} seconds')
    sorted_faces = sorted(faces, key=lambda x: (x[1]-x[3])*(x[2]-x[0]), reverse=True)
    if len(sorted_faces) > 0:
        top, right, bottom, left = sorted_faces[0]
        arr_face = arr_image[top:bottom, left:right].copy()
        pil_face = Image.fromarray(np.uint8(arr_face))
        return pil_face
    return None
    
def preprocess(image: Image.Image):
    image = image.resize(INPUT_SHAPE_2)
    image = np.expand_dims(image, 0)
    return image

def load_model(model_path):
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    return model 

def load_tf_model():
    gmodel = load_model('./model/gender_model.json')
    emodel = load_model('./model/emotion_model.json')
    gmodel.load_weights('./model/gender_weights.h5')
    emodel.load_weights('./model/emotion_weights.h5')
    print("Loaded model from disk")
    return gmodel, emodel

gmodel, emodel = load_tf_model()
df = pd.read_excel('./data/Caption.xlsx', sheet_name=[1, 2, 3, 4], header=None)

def predict(image:np.ndarray):
    pred = gmodel.predict(image)
    if pred[0] > 0.5:
        gender = 'WOMEN'
    else:
        gender = 'MEN'

    pred = emodel.predict(image)
    if pred[0] > 0.5:
        emotion = 'NORMAL'
    else:
        emotion = 'SMILE'
    
    # Generate a caption for the image
    if gender == 'WOMEN':
        if emotion == 'SMILE':
            caption = df[1]        
        else:
            caption = df[2]
    else:
        if emotion == 'SMILE':
            caption = df[3]
        else:
            caption = df[4]

    num = np.random.randint(len(caption))
    return caption.iloc[num][0]
