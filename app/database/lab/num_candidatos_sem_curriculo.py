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
PROJECT_ROOT_DIR = os.getenv("PROJECT_ROOT_DIR", "fiap-datathon")

# Infos Mongodb
MONGO_USERNAME = os.getenv("MONGO_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_ROOT_PASSWORD")
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017"

# Diretórios
PROJECT_ROOT_DIR = get_project_root(project_name=PROJECT_ROOT_DIR)

# NLP
MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L12-v2"
_ = SentenceTransformerEmbeddingStrategy(
    model_name=MODELO_EMBEDDING
)  # Garantindo o download do modelo
PRECISAO_EMBEDDING = "float32"

# Logging
configure_logging(project_name=PROJECT_ROOT_DIR, log_to_file=True, log_level="DEBUG")

# =============================================================================
# FUNÇÕES
# =============================================================================


async def algo():
    # 1. Conectando ao MongoDB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["alecrim_db"]
    collection = db["applicants"]

    #
    logger.debug(f"Without using filter")
    count = await collection.count_documents({})
    logger.debug(f"Found {count} documents")

    #
    text_field = "cv_pt"

    #
    filter = {text_field: {"$exists": True, "$nin": [None, ""]}}
    logger.debug(f"Using filter: {filter}")
    count = await collection.count_documents(filter)
    logger.debug(f"Found {count} documents matching filter")

    #
    filter = {
        text_field: {"$exists": True, "$nin": [None, ""]},
        f"embeddings.{text_field}": {"$exists": False},
    }
    logger.debug(f"Using filter: {filter}")
    count = await collection.count_documents(filter)
    logger.debug(f"Found {count} documents matching filter")


asyncio.run(algo())
