import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def index():
    return {'message':'Salut tout le monde'}

@app.get('/{name}')
def get_name(name: str):
    return {'message':f'Salut {name}'}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=3000)
    