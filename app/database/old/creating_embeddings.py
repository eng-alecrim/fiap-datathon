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
configure_logging(project_name=PROJECT_ROOT_DIR, log_to_file=True, log_level="INFO")

# =============================================================================
# FUNÇÕES
# =============================================================================

# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


def generate_bson_vector(
    vector: List[float], vector_dtype: BinaryVectorDtype = BinaryVectorDtype.FLOAT32
) -> Binary:
    return Binary.from_vector(vector, vector_dtype)


# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


async def embedding_collection_applicants(
    db_uri: str, db_name: str, collection_name: str, text_field: str
):
    # 1. Conectando ao MongoDB
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 2. Cursor para TODOS documentos que estão na coleção
    filter = {
        text_field: {"$exists": True, "$nin": [None, ""]},
        f"embeddings.{text_field}": {"$exists": False},
    }
    documents_cursor = collection.find(filter)

    # 3. Criando um for-loop
    i = 0
    bulk_size = 200
    bulk_operations = []
    async for doc in documents_cursor:
        # Infos
        doc_id = doc["_id"]
        text = doc[text_field]

        # Pré-processamento
        preprocessor = Preprocessor(NormalizationStrategy, lower=False, accent=True)
        normalized_text = preprocessor.apply(text=text)

        # Embedding
        embedder = EmbeddingModel(
            SentenceTransformerEmbeddingStrategy,
            model_name=MODELO_EMBEDDING,
            precision=PRECISAO_EMBEDDING,
        )
        text_embedding = embedder.create_embedding(text=normalized_text)

        # Gerando vetor BSON vector a partir de um embedding do tipo float32
        bson_float32_embedding = generate_bson_vector(text_embedding)
        logger.debug(f"Embedding do 'cv_pt' para '{doc_id}' feito com sucesso.")
        i += 1

        #  Criando o atributo "embedding" no documento atual e salvando a alteração
        bulk_operations.append(
            UpdateOne(
                {"_id": doc_id},
                {"$set": {f"embeddings.{text_field}": bson_float32_embedding}},
            )
        )

        # Executando as operações em lotes
        if len(bulk_operations) >= bulk_size:
            if bulk_operations:
                await collection.bulk_write(bulk_operations)
                logger.info(f"Processados {i} documentos")
                bulk_operations = []

        if i % bulk_size == 0:
            logger.info(f"Embedding do 'cv_pt' para '{doc_id}' feito com sucesso.")

    # Executando as operações restantes
    if bulk_operations:
        await collection.bulk_write(bulk_operations)
        logger.info(f"Processados {i} documentos no total")


async def embedding_collection_vagas(
    db_uri: str, db_name: str, collection_name: str, text_field: str
):
    # 1. Conectando ao MongoDB
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 2. Cursor para TODOS documentos que estão na coleção
    text_field_1 = f"{text_field[0]}.{text_field[1][0]}"
    logger.debug(f"{text_field_1 = }")
    text_field_2 = f"{text_field[0]}.{text_field[1][1]}"
    logger.debug(f"{text_field_2 = }")

    filter = {
        text_field_1: {"$exists": True, "$nin": [None, ""]},
        text_field_2: {"$exists": True, "$nin": [None, ""]},
        # f"embeddings.{text_field_1}": {"$exists": False},
        # f"embeddings.{text_field_2}": {"$exists": False},
        "embeddings.atividades_competencia_tecnicas_e_comportamentais": {
            "$exists": False
        },
    }
    documents_cursor = collection.find(filter)

    # 3. Criando um for-loop
    i = 0
    bulk_size = 200
    bulk_operations = []
    async for doc in documents_cursor:
        # Infos
        doc_id = doc["_id"]
        text = (
            doc[text_field[0]][text_field[1][0]]
            + ". "
            + doc[text_field[0]][text_field[1][1]]
        )

        # Pré-processamento
        preprocessor = Preprocessor(NormalizationStrategy, lower=False, accent=True)
        normalized_text = preprocessor.apply(text=text)

        # Embedding
        embedder = EmbeddingModel(
            SentenceTransformerEmbeddingStrategy,
            model_name=MODELO_EMBEDDING,
            precision=PRECISAO_EMBEDDING,
        )
        text_embedding = embedder.create_embedding(text=normalized_text)

        # Gerando vetor BSON vector a partir de um embedding do tipo float32
        bson_float32_embedding = generate_bson_vector(text_embedding)
        logger.debug(f"Embedding do 'cv_pt' para '{doc_id}' feito com sucesso.")
        i += 1

        #  Criando o atributo "embedding" no documento atual e salvando a alteração
        bulk_operations.append(
            UpdateOne(
                {"_id": doc_id},
                {
                    "$set": {
                        "embeddings.atividades_competencia_tecnicas_e_comportamentais": bson_float32_embedding
                    }
                },
            )
        )

        # Executando as operações em lotes
        if len(bulk_operations) >= bulk_size:
            if bulk_operations:
                await collection.bulk_write(bulk_operations)
                logger.info(f"Processados {i} documentos")
                bulk_operations = []

        if i % bulk_size == 0:
            logger.info(f"Embedding do 'cv_pt' para '{doc_id}' feito com sucesso.")

    # Executando as operações restantes
    if bulk_operations:
        await collection.bulk_write(bulk_operations)
        logger.info(f"Processados {i} documentos no total")


async def just_do_it(db_uri: str, db_name: str, collection_name: str, text_field: str):
    if collection_name == "applicants":
        await embedding_collection_applicants(
            db_uri, db_name, collection_name, text_field
        )
    elif collection_name == "vagas":
        await embedding_collection_vagas(db_uri, db_name, collection_name, text_field)


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    database_name = "alecrim_db"
    collections_and_text_fields = [
        ("applicants", "cv_pt"),
        # (
        #     "vagas",
        #     (
        #         "perfil_vaga",
        #         ["principais_atividades", "competencia_tecnicas_e_comportamentais"],
        #     ),
        # ),
    ]
    logger.info("Importando arquivos JSON para o MongoDB . . .")
    logger.debug(f"MongoDB URI: {MONGO_URI}")
    logger.debug(f"Database: {database_name}")
    logger.debug(f"Collections and text fields: {collections_and_text_fields}")
    for collection, text_field in collections_and_text_fields:
        await just_do_it(
            db_uri=MONGO_URI,
            db_name=database_name,
            collection_name=collection,
            text_field=text_field,
        )
    logger.info("Importação concluída com sucesso!")
    return None


if __name__ == "__main__":
    asyncio.run(main())
