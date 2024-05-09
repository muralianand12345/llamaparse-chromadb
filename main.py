import os
import json
import glob
import nest_asyncio
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from starlette.responses import Response
from llama_index.core import Settings

from utils.chromadb import StoreVector, LoadData, QuerySearch
from utils.openai_embed import OpenAIEmbed

nest_asyncio.apply()

env_config = dotenv_values(".env")
os.environ["LLAMA_CLOUD_API_KEY"] = env_config["LLAMA_CLOUD_API_KEY"]
os.environ["OPENAI_API_KEY"] = env_config["OPENAI_API_KEY"]

config = json.load(open("config.json"))

if not config:
    raise Exception("Config file not found")

storage_path = config.get("storage_path")
collection_name = config.get("collection_name")
result_type = config.get("result_type")

if not all([storage_path, collection_name, result_type]):
    raise Exception("Config file is missing required fields")


def read_data_folder(data_folder):
    """
    Read the data folder and return the documents path.

    Parameters:
        data_folder (str): The path to the data folder.

    Returns:
        str: The path to the documents.
    """
    documents = glob.glob(os.path.join(data_folder, "*"))
    return documents


documents_path = read_data_folder("./data")
print(documents_path)

openai_embed = OpenAIEmbed(
    embedding_model=config.get("embedding_model"),
    generator_model=config.get("generator_model"),
)

llm, embed_model = openai_embed.init_embedding()
Settings.llm = llm
Settings.embed_model = embed_model

storevector = StoreVector(
    storage_path=storage_path,
    collection_name=collection_name,
    result_type=result_type,
    documents_path=documents_path,
)


loaddata = LoadData(
    storage_path=storage_path,
    collection_name=collection_name,
    result_type=result_type,
    documents_path=documents_path,
)


searchdata = QuerySearch(storage_path=storage_path, collection_name=collection_name)

app = FastAPI()

try:
    if not os.path.exists(storage_path):
        loaddata.load_db()
    index = searchdata.load_index()
except Exception as e:
    raise Exception(str(e))


@app.get("/")
def read_root():
    """
    Check if the API is running.
    """
    return {"message": "API is running"}


@app.post("/db/reload")
def reload_db():
    """
    Reload the database with the latest data.
    """
    try:
        loaddata.load_db()
        return {"message": "Database reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def search_query(query: str):
    """
    Search the query in the database and return the response.
    """
    try:
        template = """
            You are Lee, a chatbot that can answer queries from users.
            You should be able to answer questions from the database or from the internet.
            If the question is not clear, you can ask for more information.
            User asks: "{}"
            Do not answer if the question is not safe for work.
            Reply with the answer in the format of json with the response, reference_link, image_link, and page_number.
            The reference_link should be the link to the source of the answer or the document name if it is from the database.
            Attach an image_link if the answer has an image or if possible. The image_link should be the link to the image.
            The page_number should be the page number in the document where the answer was found and should be a number.
            If the question is asked outside the database, you can add output_type as "not_in_db" in the json response.
            In your response, please use reference to the document Ideally with headings or figures.
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
