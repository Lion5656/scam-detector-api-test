from fastapi import FastAPI

app = FastAPI()

@app.get("/users/me")
def user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
def user_detail(user_id: str):
    return {"user_id": user_id}

@app.get("/files/{file_path:path}")
def file_read(file_path: str):
    return {"file_path", file_path}

@app.get("/users/{user_id}/items/{item_id}")
def read_user_item(user_id: int, item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update({"description": "description is long"})

    return item