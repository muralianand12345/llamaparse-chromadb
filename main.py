import os
import json
import nest_asyncio
from fastapi import FastAPI, HTTPException
from dotenv import dotenv_values
from starlette.responses import Response

from functions.store_vector import read_data_folder
from functions.load_data import load_db
from functions.query_search import load_index

nest_asyncio.apply()

env_config = dotenv_values(".env")
os.environ["LLAMA_CLOUD_API_KEY"] = env_config["LLAMA_CLOUD_API_KEY"]
os.environ["OPENAI_API_KEY"] = env_config["OPENAI_API_KEY"]

config = json.load(open("config.json"))

storage_path = config.get("storage_path")
collection_name = config.get("collection_name")
result_type = config.get("result_type")
documents_path = read_data_folder("./data")

app = FastAPI()

load_db(storage_path, collection_name, result_type, documents_path)
index = load_index(storage_path, collection_name)


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/db/reload")
def reload_db():
    try:
        load_db(storage_path, collection_name, result_type, documents_path)
        return {"message": "Database reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def search_query(query: str):
    try:
        template = """
            You are Lee a chatbot that can answer queries from users.
            You should be able to answer questions from the database or from the internet.
            If the question is not clear, you can ask for more information.
            User asks: "{}"
            Do not answer if the question is not safe for work.
            Reply with the answer in the format of json with the response, reference_link and image_link.
            The reference_link should be the link to the source of the answer or the document name if it is from the database.
            Attach an image_link if the answer has an image or if possible. The image_link should be the link to the image.
            If the question is asked outside the database, you can add output_type as "not_in_db" in the json response.
        """
        template = template.format(query)

        query_engine = index.as_query_engine()
        bot_response = query_engine.query(template)

        response_json = json.loads(bot_response.response)

        return Response(
            content=json.dumps(response_json), media_type="application/json"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
