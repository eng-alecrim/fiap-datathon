# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import json
import asyncio
from bson.binary import Binary
from bson import ObjectId
import multiprocessing as mp
from bson.binary import BinaryVectorDtype
import os
from typing import List, Optional
from pymongo import UpdateOne
from datetime import datetime


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

# Para lidar com a questão do uso de CUDA
mp.set_start_method("spawn", force=True)

# =============================================================================
# CLASSES
# =============================================================================


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


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


def process_embeddings(texts_with_ids):
    """Process a batch of texts to create embeddings"""
    # Initialize the models (once per process)
    preprocessor = Preprocessor(NormalizationStrategy, lower=False, accent=True)
    embedder = EmbeddingModel(
        SentenceTransformerEmbeddingStrategy,
        model_name=MODELO_EMBEDDING,
        precision=PRECISAO_EMBEDDING,
    )

    results = []
    for item in texts_with_ids:
        doc_id = item["id"]
        text = item["text"]

        try:
            # Preprocessing
            normalized_text = preprocessor.apply(text=text)

            # Embedding generation
            text_embedding = embedder.create_embedding(text=normalized_text)

            # Generate BSON vector
            bson_float32_embedding = generate_bson_vector(text_embedding)

            # Store result
            results.append(
                {"id": doc_id, "embedding": bson_float32_embedding, "success": True}
            )
        except Exception as e:
            # Log any errors but continue processing other items
            results.append({"id": doc_id, "error": str(e), "success": False})

    return results


# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


async def fetch_documents_in_batches(
    collection, filter_query, text_field, batch_size=100
):
    """Fetch documents in batches with retry logic"""
    cursor = collection.find(filter_query).limit(
        500
    )  # TODO REMOVER O LIMITE AQUI DEPOIS!
    batch = []
    total_processed = 0

    try:
        async for doc in cursor:
            # Extract only what we need for processing
            item = {"id": str(doc["_id"]), "text": doc[text_field]}
            batch.append(item)

            if len(batch) >= batch_size:
                yield batch
                total_processed += len(batch)
                logger.info(
                    f"Fetched batch of {len(batch)} documents, total so far: {total_processed}"
                )
                batch = []
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        # If there was an error but we have some documents, still yield them
        if batch:
            yield batch

    # Don't forget remaining documents
    if batch:
        yield batch
        logger.info(f"Fetched final batch of {len(batch)} documents")


# -----------------------------------------------------------------------------
# List[float] -> Binary
# -----------------------------------------------------------------------------


async def embedding_collection_applicants_parallel(
    db_uri: str,
    db_name: str,
    collection_name: str,
    text_field: str,
    num_processes: Optional[int] = None,
    batch_size: int = 100,
    bulk_write_size: int = 200,
):
    """Process embeddings in parallel using multiprocessing while keeping DB operations in main process"""
    # Use default number of CPU cores if not specified
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() * 6 // 10)  # Leave one core free

    logger.info(f"Starting parallel processing with {num_processes} processes")

    # 1. Connect to MongoDB from the main process only
    client = AsyncIOMotorClient(db_uri)
    db = client[db_name]
    collection = db[collection_name]

    # 2. Create filter for documents that need embedding
    filter_query = {
        text_field: {"$exists": True, "$nin": [None, ""]},
        f"embeddings.{text_field}": {"$exists": False},
    }

    # 3. Create a process pool - only for embedding calculations
    with mp.Pool(processes=num_processes) as pool:
        total_processed = 0
        bulk_operations = []

        # 4. Process documents in batches
        async for doc_batch in fetch_documents_in_batches(
            collection, filter_query, text_field, batch_size
        ):
            if not doc_batch:
                continue

            # Process batch in parallel pool - send only the text data, not DB connections
            try:
                # Use apply_async with a callback to avoid blocking
                batch_results = await asyncio.to_thread(
                    pool.apply,  # Using apply for simplicity, apply_async for more advanced use
                    process_embeddings,
                    (doc_batch,),
                )

                # Create bulk update operations from results
                for result in batch_results:
                    if result["success"]:
                        doc_id = ObjectId(result["id"])  # Convert back to ObjectId
                        embedding = result["embedding"]

                        bulk_operations.append(
                            UpdateOne(
                                {"_id": doc_id},
                                {"$set": {f"embeddings.{text_field}": embedding}},
                            )
                        )
                    else:
                        logger.warning(
                            f"Failed to process document {result['id']}: {result.get('error')}"
                        )

                # Perform bulk write once we reach the specified size
                if len(bulk_operations) >= bulk_write_size:
                    try:
                        await collection.bulk_write(bulk_operations)
                        total_processed += len(bulk_operations)
                        logger.info(f"Updated {total_processed} documents in database")
                        bulk_operations = []
                    except Exception as e:
                        logger.error(f"Error during bulk write: {e}")
                        # Re-establish connection if needed
                        client = AsyncIOMotorClient(db_uri)
                        db = client[db_name]
                        collection = db[collection_name]
                        # Clear the failed operations to avoid repeated failures
                        bulk_operations = []

            except Exception as e:
                logger.error(f"Error during batch processing: {e}")

    # 5. Process any remaining bulk operations
    if bulk_operations:
        try:
            await collection.bulk_write(bulk_operations)
            total_processed += len(bulk_operations)
            logger.info(f"Updated final batch of {len(bulk_operations)} documents")
        except Exception as e:
            logger.error(f"Error during final bulk write: {e}")

    logger.info(f"Completed processing {total_processed} documents in total")
    return total_processed


# =============================================================================
# List[float] => Binary
# =============================================================================


async def main():
    db_uri = MONGO_URI
    db_name = "alecrim_db"
    collection_name = "applicants"
    text_field = "cv_pt"

    try:
        # Process with multiprocessing, but leave one core free for system operations
        total_processed = await embedding_collection_applicants_parallel(
            db_uri=db_uri,
            db_name=db_name,
            collection_name=collection_name,
            text_field=text_field,
            num_processes=max(1, mp.cpu_count() * 6 // 10),
            batch_size=100,  # Smaller batch size to avoid memory issues
            bulk_write_size=200,
        )

        logger.info(f"Successfully processed {total_processed} documents")
    except Exception as e:
        logger.error(f"Error in main process: {e}")


if __name__ == "__main__":
    # Set up proper shutdown for multiprocessing
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
