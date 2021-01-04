# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile
from prediction import read_image, preprocess, predict, detect_face
# from prediction import read_image, preprocess, predict
import time 

# 2. Create the app object
app = FastAPI()

@app.get('/')
def index():
    '''
    This is a first docstring.
    '''
    return {'message': 'Hello, WORLD'}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    '''
    Predict gender of people in picture.
    '''
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    start = time.time()
 
    image = read_image(await file.read())
    face = detect_face(image)
    # face = image 
    
    prediction = None 
    if face is not None:
        face = preprocess(face)
        prediction = predict(face)
        end = time.time()
        print(f"Total prediction time : {end-start:.2f} seconds.")
    return prediction

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)