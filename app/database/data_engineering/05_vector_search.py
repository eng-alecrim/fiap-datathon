# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import find_dotenv, load_dotenv
import os
from common.utils import get_project_root
from common.logging import configure_logging
from custom_nlp.embedding import EmbeddingModel, SentenceTransformerEmbeddingStrategy
from custom_nlp.preprocessing import NormalizationStrategy, Preprocessor
from loguru import logger
import asyncio


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

# Tratamento texto
preprocessor = Preprocessor(NormalizationStrategy, lower=False, accent=True)

# Embedding
MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L12-v2"
PRECISAO_EMBEDDING = "float32"
embedder = EmbeddingModel(
    SentenceTransformerEmbeddingStrategy,
    model_name=MODELO_EMBEDDING,
    precision=PRECISAO_EMBEDDING,
)  # Garantindo o download do modelo

# Diretórios
PROJECT_ROOT_DIR = get_project_root(project_name=PROJECT_NAME)

# Logging
configure_logging(project_name=PROJECT_NAME, log_to_file=True, log_level="INFO")

# =============================================================================
# MAIN
# =============================================================================


async def main() -> None:  # Changed to async function
    collection_name = "applicants"

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DATABASE]
    collection = db[collection_name]

    texto = """Engenheiro mecânico é o profissional responsável por projetar, desenvolver, supervisionar e manter sistemas e equipamentos mecânicos, como máquinas, motores, veículos e instalações industriais. Atua em diversas áreas, incluindo indústria automotiva, aeroespacial, energia, manufatura e tecnologia, sempre buscando eficiência, segurança e inovação nos processos e produtos."""
    embedding = embedder.create_embedding(text=preprocessor.apply(texto)).tolist()

    cursor = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embeddings.cv_pt",
                    "queryVector": embedding,
                    "numCandidates": 50,
                    "limit": 5,
                }
            },
            ## We are extracting 'vectorSearchScore' here
            ## columns with 1 are included, columns with 0 are excluded
            {
                "$project": {
                    "_id": 1,
                    "infos_basicas.nome": 1,
                    "cv_pt": 1,
                    "search_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
    )

    results_as_list = await cursor.to_list(length=5)

    for result in results_as_list:
        print(result)

    return None


if __name__ == "__main__":
    # Set up proper shutdown for multiprocessing
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    # except Exception as e:
    #     logger.error(f"Unhandled exception: {e}")
