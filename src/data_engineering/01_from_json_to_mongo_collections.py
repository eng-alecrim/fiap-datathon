# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import asyncio
import json
import os
from pathlib import Path

from common.utils import get_project_root
from common.logging import configure_logging
from dotenv import find_dotenv, load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from loguru import logger

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

project_name = os.getenv("PROJECT_NAME", "fiap-datathon")

adm_username = os.getenv("MONGODB_INITDB_ROOT_USERNAME")
adm_password = os.getenv("MONGODB_INITDB_ROOT_PASSWORD")
mongo_uri = (
    f"mongodb://{adm_username}:{adm_password}@localhost:27017/?directConnection=true"
)

dir_projeto = get_project_root(project_name=project_name)

configure_logging(project_name=project_name, log_to_file=True, log_level="DEBUG")

# =============================================================================
# FUNÇÕES
# =============================================================================

# -----------------------------------------------------------------------------
# JSON -> MongoDB
# -----------------------------------------------------------------------------


async def import_json_to_mongodb(
    json_file_path: Path, db_uri: str, db_name: str, collection_name: str
) -> None:
    # 1. Conectando ao MongoDB
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]

    # 2. Verifica se a coleção já existe
    existing_collections = await db.list_collection_names()
    if collection_name in existing_collections:
        logger.debug(f"A coleção '{collection_name}' já existe.")
        return None

    logger.debug(f"Criando a coleção '{collection_name}' . . .")
    collection = db[collection_name]

    # 3. Lendo o arquivo JSON
    with open(json_file_path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)

    # 4. Transformação dict -> list[Documents]
    documents = []
    for id_key, attributes in data_dict.items():
        # Copiando para evitar sobrescrita
        document = attributes.copy()
        # Add o ID original como um campo
        document["id"] = id_key
        # Add à lista de documentos
        documents.append(document)

    # 5. Inserindo os documentos na coleção
    if documents:
        result = await collection.insert_many(documents)
        logger.debug(f"{len(result.inserted_ids)} documentos inseridos com sucesso!")
        return None

    logger.debug("Nenhum documento encontrado.")

    return None


# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:
    database_name = "alecrim_db"
    collections = ["applicants", "vagas", "prospects"]
    logger.info("Importando arquivos JSON para o MongoDB . . .")
    logger.debug(f"MongoDB URI: {mongo_uri}")
    logger.debug(f"Database: {database_name}")
    logger.debug(f"Collections: {collections}")
    for collection in collections:
        json_file = dir_projeto / f"data/raw/{collection}.json"
        logger.debug(f"Importando arquivo JSON: {json_file}")
        await import_json_to_mongodb(json_file, mongo_uri, database_name, collection)
        logger.debug(f"Arquivo JSON importado com sucesso: {json_file}")
    logger.info("Importação concluída com sucesso!")
    return None


if __name__ == "__main__":
    asyncio.run(main())
