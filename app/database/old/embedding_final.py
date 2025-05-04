# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from functools import partial
import asyncio
from bson.binary import Binary
import multiprocessing as mp
from bson.binary import BinaryVectorDtype
import os
from typing import List, Dict, Any, Optional
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

# Para lidar com a questão do uso de CUDA
mp.set_start_method("spawn", force=True)

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


def process_batch(batch_docs: List[Dict[str, Any]], text_field: str) -> List[Dict]:
    results = []

    # Create embedding model once per process (more efficient)
    preprocessor = Preprocessor(NormalizationStrategy, lower=False, accent=True)
    embedder = EmbeddingModel(
        SentenceTransformerEmbeddingStrategy,
        model_name=MODELO_EMBEDDING,
        precision=PRECISAO_EMBEDDING,
    )

    for doc in batch_docs:
        doc_id = doc["_id"]
        text = doc[text_field]

        # Preprocessing
        normalized_text = preprocessor.apply(text=text)

        # Embedding generation
        text_embedding = embedder.create_embedding(text=normalized_text)

        # Generate BSON vector
        bson_float32_embedding = generate_bson_vector(text_embedding)

        # Store result
        results.append({"doc_id": doc_id, "embedding": bson_float32_embedding})

    return results


# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


async def fetch_documents(collection, filter_query, batch_size=1000):
    """Fetch documents in batches to avoid loading everything into memory at once"""
    cursor = collection.find(filter_query)
    current_batch = []

    async for doc in cursor:
        current_batch.append(doc)

        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch


# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


async def embedding_collection_applicants_parallel(
    db_uri: str,
    db_name: str,
    collection_name: str,
    text_field: str,
    num_processes: Optional[int] = None,
    batch_size: int = 1000,
    bulk_write_size: int = 200,
):
    """Process embeddings in parallel using multiprocessing"""
    # Use default number of CPU cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()

    # 1. Connect to MongoDB
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 2. Create filter for documents that need embedding
    filter_query = {
        text_field: {"$exists": True, "$nin": [None, ""]},
        f"embeddings.{text_field}": {"$exists": False},
    }

    # 3. Create a process pool
    pool = mp.Pool(processes=num_processes)
    process_func = partial(process_batch, text_field=text_field)

    total_processed = 0
    bulk_operations = []

    # 4. Process documents in batches
    async for doc_batch in fetch_documents(collection, filter_query, batch_size):
        # Process batch in parallel
        batch_results = await asyncio.to_thread(pool.apply, process_func, (doc_batch,))

        # Create bulk update operations from results
        for result in batch_results:
            doc_id = result["doc_id"]
            embedding = result["embedding"]

            bulk_operations.append(
                UpdateOne(
                    {"_id": doc_id},
                    {"$set": {f"embeddings.{text_field}": embedding}},
                )
            )

            # Perform bulk write once we reach the specified size
            if len(bulk_operations) >= bulk_write_size:
                await collection.bulk_write(bulk_operations)
                total_processed += len(bulk_operations)
                logger.info(f"Processed {total_processed} documents")
                bulk_operations = []

    # 5. Process any remaining bulk operations
    if bulk_operations:
        await collection.bulk_write(bulk_operations)
        total_processed += len(bulk_operations)
        logger.info(f"Processed {total_processed} documents in total")

    # 6. Close the process pool
    pool.close()
    pool.join()

    return total_processed


# Example usage
async def main():
    db_uri = MONGO_URI
    db_name = "alecrim_db"
    collection_name = "applicants"
    text_field = "cv_pt"

    # Use all available CPU cores
    total_processed = await embedding_collection_applicants_parallel(
        db_uri=db_uri,
        db_name=db_name,
        collection_name=collection_name,
        text_field=text_field,
        num_processes=mp.cpu_count(),  # Adjust this based on your system
        batch_size=1000,  # How many documents to fetch at once
        bulk_write_size=200,  # How many documents to update in each bulk write
    )

    print(f"Total documents processed: {total_processed}")


if __name__ == "__main__":
    asyncio.run(main())
