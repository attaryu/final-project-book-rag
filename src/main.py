from fastapi import FastAPI
from enum import Enum

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World!"}


@app.get("/greeting/{name}")
def greeting(name: str):
    return {"message": "Hello " + name.title()}


class FooBarEnum(str, Enum):
    foo = "foo"
    bar = "bar"


@app.get("/test/{foo_bar_enum}")
def foo_bar_test(foo_bar_enum: FooBarEnum):
    init_message = "I'm " + foo_bar_enum + ", welcome "

    if foo_bar_enum is FooBarEnum.bar:
        return {"message": init_message + "foo!"}

    return {"message": init_message + "bar!"}


@app.get("/public/{file_path:path}")
def get_public_path(file_path: str):
    return {"message": "You accessed " + file_path + " path!"}
