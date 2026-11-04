from fastapi import FastAPI
from login import login

app=FastAPI()

@app.get("/in")
def find():
    dict_=login()
    return {'No.of jobs':len(dict_),'details':dict_}


@app.get("/")
def find():
    return "hello"