import asyncio
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCursor,
)
from typing import List, Dict, Any
from beanie import Document, init_beanie
from bson import ObjectId
import os
from dotenv import find_dotenv, load_dotenv


# Configuração do projeto
load_dotenv(find_dotenv())
adm_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
adm_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
mongo_uri = f"mongodb://{adm_username}:{adm_password}@localhost:27017"


#
class DocUsuario(Document):
    nome: str
    idade: int

    class Settings:
        name = "temp_collection"
        use_state_management = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "nome": "alecrim",
                "idade": "23",
            }
        }


async def init(
    mongo_uri: str, collection_name: str, document_models: List[Document]
) -> AsyncIOMotorDatabase:
    """Inicializa Beanie e retorna o objeto de banco de dados motor."""
    client = AsyncIOMotorClient(mongo_uri)
    db = client[collection_name]
    await init_beanie(database=db, document_models=document_models)
    print("Conexão com Beanie inicializada.")
    return db


async def get_batch_from_cursor(
    cursor: AsyncIOMotorCursor, batch_size: int
) -> List[Dict[str, Any]]:
    """Lê um lote de documentos de um cursor motor."""
    batch = []
    i = 0
    for i in range(batch_size):
        try:
            while i < batch_size:
                async for doc in cursor:
                    batch.append(doc)
                    i += 1
        except StopAsyncIteration:
            break  # Cursor esgotado
    print("batch!")
    return batch


async def get_batch_from_cursor_corrected(
    cursor: AsyncIOMotorCursor, batch_size: int
) -> List[Dict[str, Any]]:
    """Lê um lote de documentos de um cursor motor (versão corrigida)."""
    batch = []
    # Garante que não tentaremos ler um batch de tamanho 0 ou negativo
    if batch_size <= 0:
        return batch

    try:
        # Itera sobre o cursor de forma assíncrona
        async for doc in cursor:
            batch.append(doc)
            # Verifica se o batch atingiu o tamanho desejado APÓS adicionar
            if len(batch) == batch_size:
                break  # Sai do loop async for pois o batch está cheio
        # O loop async for termina naturalmente se o cursor for esgotado
        # antes de atingir batch_size. A exceção StopAsyncIteration
        # é tratada implicitamente pelo `async for`.

    except Exception as e:
        # É bom capturar outras exceções que podem ocorrer durante a iteração
        # (problemas de rede, erros do Motor/MongoDB, etc.)
        print(f"Erro durante a iteração do cursor: {e}")
        # Dependendo do caso, você pode querer retornar o batch parcial
        # ou levantar a exceção novamente: raise e

    print(f"Batch retornado! Tamanho: {len(batch)}")  # Print mais informativo
    return batch


async def main() -> None:
    # Criando um cursor
    document_models = [DocUsuario]
    cursor = await init(
        mongo_uri=mongo_uri,
        collection_name="alecrim_db",
        document_models=document_models,
    )
    temp_collection = cursor["temp_collection"]
    # Aqui tá tudo
    algo = temp_collection.find({})

    # Lendo um lote de documentos
    batch_size = 2

    for _ in range(batch_size):
        batch = await get_batch_from_cursor_corrected(algo, batch_size)
        print(batch)
        break

    # async for doc in algo:
    #     print(doc)

    return None


if __name__ == "__main__":
    asyncio.run(main())
