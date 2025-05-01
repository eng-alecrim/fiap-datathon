# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from motor.motor_asyncio import AsyncIOMotorClient
import time  # Changed from 'from time import time' to import the entire module
from dotenv import find_dotenv, load_dotenv
import os
from common.utils import get_project_root
from common.logging import configure_logging
from loguru import logger
import asyncio
from pymongo.operations import SearchIndexModel


load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

# Nome do Projeto
PROJECT_NAME = os.getenv("PROJECT_NAME", "fiap-datathon")

# Infos Mongodb
MONGO_USERNAME = os.getenv("MONGODB_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGODB_INITDB_ROOT_PASSWORD")
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27017/?directConnection=true"

# Diretórios
PROJECT_DIR = get_project_root(project_name=PROJECT_NAME)

# Logging
configure_logging(project_name=PROJECT_NAME, log_to_file=True, log_level="INFO")

# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:  # Changed to async function
    db_name = "alecrim_db"
    collection_name = "applicants"

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[db_name]
    collection = db[collection_name]

    # Create your index model, then create the search index
    search_index_model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embeddings.cv_pt",
                    "numDimensions": 384,
                    "similarity": "dotProduct",
                    "quantization": "scalar",
                }
            ]
        },
        name="vector_index",
        type="vectorSearch",
    )

    result = await collection.create_search_index(model=search_index_model)
    print(result)
    index_name = result  # Extract index name from result

    print(
        f"Polling to check if the index {index_name} is ready. This may take up to a minute."
    )
    while True:
        # Properly await and convert cursor to list
        indices = await collection.list_search_indexes().to_list(length=None)

        # Check if the index is ready (queryable)
        index_ready = False
        for index in indices:
            if index.get("name") == index_name and index.get("queryable") is True:
                index_ready = True
                break

        if index_ready:
            break

        print("Index not ready yet, waiting 5 seconds...")
        time.sleep(5)

    print(f"{index_name} is ready for querying.")

    return None


if __name__ == "__main__":
    # Set up proper shutdown for multiprocessing
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
