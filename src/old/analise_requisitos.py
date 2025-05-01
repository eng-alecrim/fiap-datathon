# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import asyncio
import os
from datetime import datetime
from typing import Tuple

from beanie import Document, init_beanie
from bson import ObjectId
from dotenv import find_dotenv, load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo import MongoClient

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

adm_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
adm_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
mongo_uri = f"mongodb://{adm_username}:{adm_password}@localhost:27017"

client = MongoClient(mongo_uri)
db = client["alecrim_db"]
collection_vagas = db["vagas"]
collection_applicants = db["applicants"]
collection_prospects = db["prospects"]

niveis_idioma = {
    nivel: num
    for num, nivel in enumerate([
        "Nenhum",
        "Básico",
        "Intermediário",
        "Avançado",
        "Fluente",
    ])
}

niveis_ensino = {
    nivel: num
    for num, nivel in enumerate([
        # Fundamental
        "Ensino Fundamental Incompleto",
        "Ensino Fundamental Cursando",
        "Ensino Fundamental Completo",
        # Médio
        "Ensino Médio Incompleto",
        "Ensino Médio Cursando",
        "Ensino Médio Completo",
        # Técnico
        "Ensino Técnico Incompleto",
        "Ensino Técnico Cursando",
        "Ensino Técnico Completo",
        # Superior
        "Ensino Superior Incompleto",
        "Ensino Superior Cursando",
        "Ensino Superior Completo",
        # Pós
        "Pós Graduação Incompleto",
        "Pós Graduação Cursando",
        "Pós Graduação Completo",
        # Mestrado
        "Mestrado Incompleto",
        "Mestrado Cursando",
        "Mestrado Completo",
        # Doutorado
        "Doutorado Incompleto",
        "Doutorado Cursando",
        "Doutorado Completo",
    ])
}

niveis_profissional = {
    nivel: num
    for num, nivel in enumerate([
        "Aprendiz",
        "Técnico de Nível Médio",
        "Trainee",
        "Auxiliar",
        "Assistente",
        "Analista",
        "Júnior",
        "Pleno",
        "Sênior",
        "Especialista",
        "Líder",
        "Supervisor",
        "Coordenador",
        "Gerente",
    ])
}

# =============================================================================
# CLASSES
# =============================================================================


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class DocumentAnaliseRequisito(Document):
    id_vaga: str
    id_candidato: str
    desclassificado: bool
    motivos: str
    observacoes: str
    data_analise: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "analise_requisitos"  # Nome da coleção no MongoDB
        use_state_management = True

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "id_vaga": "67f72d6567764023fb9430cf",
                "id_candidato": "67f85da50bd7423e7418ff89",
                "desclassificado": False,
                "motivos": "",
                "observacoes": "Candidato não possui dados sobre PCD.",
                "data_analise": "2025-04-13 21:56:08.340104",
            }
        }


class Requisitos(BaseModel):
    pcd: str
    nivel_academico: str
    nivel_ingles: str
    nivel_espanhol: str


# =============================================================================
# FUNÇÕES
# =============================================================================


def gerador_requisitos_para_vaga(vaga: dict):
    return Requisitos(
        pcd=vaga["perfil_vaga"]["vaga_especifica_para_pcd"],
        nivel_academico=vaga["perfil_vaga"]["nivel_academico"],
        nivel_ingles=vaga["perfil_vaga"]["nivel_ingles"],
        nivel_espanhol=vaga["perfil_vaga"]["nivel_espanhol"],
    )


def gerador_requisitos_para_candidato(candidato: dict):
    return Requisitos(
        pcd=candidato["informacoes_pessoais"]["pcd"],
        nivel_academico=candidato["formacao_e_idiomas"]["nivel_academico"],
        nivel_ingles=candidato["formacao_e_idiomas"]["nivel_ingles"],
        nivel_espanhol=candidato["formacao_e_idiomas"]["nivel_espanhol"],
    )


def desclassificador(
    requisitos_vaga: Requisitos, requisitos_candidato: Requisitos
) -> Tuple[bool, str, str]:
    observacoes = []
    motivos = []
    desclassificar = False

    # 1. PCD
    if requisitos_vaga.pcd:
        if requisitos_candidato.pcd:
            if (requisitos_vaga.pcd == "Sim") & (requisitos_candidato.pcd == "Não"):
                desclassificar = True
                motivos.append("Vaga para PCD.")
        else:
            observacoes.append("Candidato não possui dados sobre PCD.")

    # 2. Nível acadêmico
    if requisitos_vaga.nivel_academico:
        if requisitos_candidato.nivel_academico:
            if niveis_ensino.get(requisitos_vaga.nivel_academico) > niveis_ensino.get(
                requisitos_candidato.nivel_academico
            ):
                desclassificar = True
                motivos.append(
                    f"Vaga requer '{requisitos_vaga.nivel_academico}', candidato tem '{requisitos_candidato.nivel_academico}'."
                )
        else:
            observacoes.append("Candidato não possui dados sobre formação acadêmica.")

    # 3. Nível idiomas
    if requisitos_vaga.nivel_ingles:
        if requisitos_candidato.nivel_ingles:
            if niveis_idioma.get(requisitos_vaga.nivel_ingles) > niveis_idioma.get(
                requisitos_candidato.nivel_ingles
            ):
                desclassificar = True
                motivos.append(
                    f"Vaga requer inglês '{requisitos_vaga.nivel_ingles.lower()}', candidato tem '{requisitos_candidato.nivel_ingles.lower()}'."
                )
        else:
            observacoes.append("Candidato não possui dados sobre nível de inglês.")

    if requisitos_vaga.nivel_espanhol:
        if requisitos_candidato.nivel_espanhol:
            if niveis_idioma.get(requisitos_vaga.nivel_espanhol) > niveis_idioma.get(
                requisitos_candidato.nivel_espanhol
            ):
                desclassificar = True
                motivos.append(
                    f"Vaga requer espanhol '{requisitos_vaga.nivel_ingles.lower()}', candidato tem '{requisitos_candidato.nivel_espanhol.lower()}'."
                )
        else:
            observacoes.append("Candidato não possui dados sobre nível de espanhol.")

    return desclassificar, " ".join(motivos), " ".join(observacoes)


async def init():
    client = AsyncIOMotorClient(mongo_uri)
    await init_beanie(
        database=client["alecrim_db"], document_models=[DocumentAnaliseRequisito]
    )


async def populate_analises(collection_vagas, collection_applicants):
    vagas = collection_vagas.find()
    for vaga in vagas:
        requisitos_vaga = gerador_requisitos_para_vaga(vaga)
        candidatos = collection_applicants.find()
        for candidato in candidatos:
            existing = await DocumentAnaliseRequisito.find_one(
                DocumentAnaliseRequisito.id_vaga == vaga["id"],
                DocumentAnaliseRequisito.id_candidato == candidato["id"],
            )
            if not existing:
                requisitos_candidato = gerador_requisitos_para_candidato(candidato)
                desclassificado, motivos, observacoes = desclassificador(
                    requisitos_vaga, requisitos_candidato
                )
                doc = DocumentAnaliseRequisito(
                    id_vaga=vaga["id"],
                    id_candidato=candidato["id"],
                    desclassificado=desclassificado,
                    motivos=motivos,
                    observacoes=observacoes,
                )
                await doc.insert()


# =============================================================================
# MAIN
# =============================================================================


async def main():
    await init()
    await populate_analises(collection_vagas, collection_applicants)


if __name__ == "__main__":
    asyncio.run(main())
