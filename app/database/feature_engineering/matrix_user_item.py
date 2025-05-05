# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import pandas as pd
import asyncio
from collections.abc import Iterable, Iterator
from functools import reduce
import os
from typing import Any, Dict
from common.logging import configure_logging
from common.utils import get_project_root
from dotenv import find_dotenv, load_dotenv
import pymongo
from beanie import init_beanie, Document, Indexed
from beanie.odm.fields import PydanticObjectId
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# CONSTANTES
# =============================================================================

load_dotenv(find_dotenv())

# Nome do Projeto
PROJECT_NAME = os.getenv("PROJECT_NAME", "fiap-datathon")

# Infos Mongodb
MONGO_USERNAME = os.getenv("MONGODB_INITDB_ROOT_USERNAME", "")
MONGO_PASSWORD = os.getenv("MONGODB_INITDB_ROOT_PASSWORD", "")
MONGO_DATABASE = os.getenv("MONGODB_DB_NAME", "")
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017/?directConnection=true"

# Diretórios
PROJECT_ROOT_DIR = get_project_root(project_name=PROJECT_NAME)

# Logging
configure_logging(project_name=PROJECT_NAME, log_to_file=True, log_level="INFO")

# =============================================================================
# MAIN
# =============================================================================


async def init_db() -> None:

    client = AsyncIOMotorClient(MONGO_URI)
    collection = client[MONGO_DATABASE]["prospects"]

    # Retrieve documents where "prospects" isn't empty
    filter = {"prospects": {"$ne": []}}
    docs = await collection.find(filter).to_list(
        length=None
    )  # .batch_size(batch_size=100)

    vaga_candidatos = {}
    f_reduce = lambda old, new: old | {new["codigo"]: 1}
    for doc in docs:
        prospects = reduce(f_reduce, doc["prospects"], {})
        vaga_candidatos[doc["id"]] = prospects

    df = (
        pd.DataFrame.from_dict(data=vaga_candidatos, orient="index")
        # .fillna(0)
        # .astype(int)
    )
    df.to_parquet("temp.parquet")  # , index=False)

    # with ThreadPoolExecutor() as executor:
    #     async for document in cursor:
    #         await asyncio.get_event_loop().run_in_executor(
    #             executor, process_item, document
    #         )
    # with multiprocessing.Pool() as pool:
    #     pool.apply(process_item, results_batches)

    return None


if __name__ == "__main__":
    asyncio.run(init_db())
