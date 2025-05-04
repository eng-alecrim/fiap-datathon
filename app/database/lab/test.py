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
    for _ in range(batch_size):
        try:
            async for doc in cursor:
                batch.append(doc)
        except StopAsyncIteration:
            break  # Cursor esgotado
    return batch


async def main() -> None:
    # Criando um cursor
    document_models = [DocUsuario]
    cursor = await init(
        mongo_uri=mongo_uri,
        collection_name="alecrim_db",
        document_models=document_models,
    )

    usuarios = [
        {"nome": "alecrim", "idade": 23},
        {"nome": "biel", "idade": 12},
        {"nome": "vinc", "idade": 10},
    ]

    for usuario in usuarios:
        new_user = DocUsuario(**usuario)
        existing = await DocUsuario.find_one(DocUsuario.nome == usuario["nome"])
        if existing:
            print(f"Usuário {usuario['nome']} já existe.")
        else:
            print(f"Usuário {usuario['nome']} não existe, inserindo...")
            await new_user.insert()
            print(f"Usuário {usuario['nome']} inserido com sucesso.")

    return None


if __name__ == "__main__":
    asyncio.run(main())
