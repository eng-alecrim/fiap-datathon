import asyncio

from collections.abc import Iterable
from typing import Optional

import pymongo
from pydantic import BaseModel

from beanie import init_beanie, Document, Indexed
from beanie.odm.fields import PydanticObjectId

from motor.motor_asyncio import AsyncIOMotorClient

from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

# Nome do Projeto
PROJECT_NAME = os.getenv("PROJECT_NAME", "fiap-datathon")

# Infos Mongodb
MONGO_USERNAME = os.getenv("MONGODB_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGODB_INITDB_ROOT_PASSWORD")
MONGO_DATABASE = os.getenv("MONGODB_DB_NAME")
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017/?directConnection=true"

# =============================================================================
# CLASSES
# =============================================================================


class SimilarityMethod(BaseModel):
    name: str
    similarity: float


class Similarity(Document):
    between: str
    id_A: PydanticObjectId
    id_B: PydanticObjectId
    methods: Iterable[SimilarityMethod]

    class Settings:
        name = "similarities"
        indexes = [
            pymongo.IndexModel(
                [
                    ("between", pymongo.ASCENDING),
                    ("id_A", pymongo.ASCENDING),
                    ("id_B", pymongo.ASCENDING),
                ],
                name="default",
            )
        ]


# =============================================================================
# MAIN
# =============================================================================


async def init_db() -> None:
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DATABASE]

    await init_beanie(database=db, document_models=[Similarity])

    print("Coleção 'similarities' criada com sucesso.")

    return None


if __name__ == "__main__":
    asyncio.run(init_db())
