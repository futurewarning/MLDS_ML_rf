import pandas as pd
import json
from fastapi import FastAPI, Request,File, UploadFile
from fastapi.templating import Jinja2Templates
import pickle
import pipeline

# Это явно костыль, но я пока хз почему не работает в нормальном режиме. Жди апдейта в лс :)
pipeline.main()
process_pipeline = pickle.load(open('models/process_pipeline.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

app = FastAPI()
templates = Jinja2Templates(directory='templates')

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse('core.html',
                                      {'request': request})

@app.post("/upload_file")
async def upload_file(request: Request,
                      file: UploadFile = File(...)):
    file_type = file.content_type[file.content_type.find('/') + 1:]
    if file_type == 'json':
        data = pd.DataFrame(json.load(file.file), index=[0])
    elif file_type == 'csv':
        data = pd.read_csv(file.file)
    else:
        return {'Result': 'File extension incorrect'}

    data = process_pipeline.transform(data)
    if 'selling_price' in data.columns:
        data.drop('selling_price', axis=1, inplace=True)
    result = model.predict(data)
    print(model.get_params())

    return_result = {}
    for i in range(len(result)):
        return_result[i] = result[i]

    return return_result

