# %%
# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import asyncio
from bson.binary import Binary
from bson.binary import BinaryVectorDtype
import os
from typing import List
from pymongo import UpdateOne


from custom_nlp.preprocessing import NormalizationStrategy, Preprocessor
from custom_nlp.embedding import SentenceTransformerEmbeddingStrategy, EmbeddingModel
from common.utils import get_project_root
from common.logging import configure_logging
from dotenv import find_dotenv, load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

# Nome do Projeto
PROJECT_NAME = os.getenv("PROJECT_NAME", "fiap-datathon")

# Infos Mongodb
MONGO_USERNAME = os.getenv("MONGO_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_ROOT_PASSWORD")
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017"

# Diretórios
PROJECT_DIR = get_project_root(project_name=PROJECT_NAME)

# NLP
MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L12-v2"
_ = SentenceTransformerEmbeddingStrategy(
    model_name=MODELO_EMBEDDING
)  # Garantindo o download do modelo
PRECISAO_EMBEDDING = "float32"

# Logging
configure_logging(project_name=PROJECT_NAME, log_to_file=True, log_level="INFO")

# %%
database_name = "alecrim_db"
collection = "vagas"
text_field = "perfil_vaga.principais_atividades"

db_uri = MONGO_URI
db_name = database_name
collection_name = collection
text_field = text_field
client = AsyncIOMotorClient(db_uri)

db = client[db_name]
collection = db[collection_name]

# %%
# 2. Cursor para TODOS documentos que estão na coleção
filter = {
    text_field: {"$exists": True, "$nin": [None, ""]},
    f"embeddings.{text_field}": {"$exists": False},
}
documents_cursor = collection.find(filter)


# %%
async def algo():
    async for doc in documents_cursor:
        print(doc)
        break


# %%
async def teste():
    await algo()


asyncio.run(teste())

# %%
