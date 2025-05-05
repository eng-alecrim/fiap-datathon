# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import asyncio
import os
from datetime import datetime
from typing import List, Optional, Set, Tuple, Dict, Any  # Adicionado Dict, Any

from beanie import Document, init_beanie
from bson import ObjectId
from dotenv import find_dotenv, load_dotenv
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCursor,
)  # Adicionado AsyncIOMotorCursor
from pydantic import BaseModel, Field
from common.logging import configure_logging
from loguru import logger
from dotenv import find_dotenv, load_dotenv
import os
from time import time

load_dotenv(find_dotenv())
project_name = os.getenv("PROJECT_NAME", "fiap-datathon")

configure_logging(project_name=project_name, log_to_file=True, log_level="DEBUG")

load_dotenv(find_dotenv())

# =============================================================================
# CONSTANTES
# =============================================================================

# Configuração do projeto
adm_username = os.getenv("MONGODB_INITDB_ROOT_USERNAME")
adm_password = os.getenv("MONGODB_INITDB_ROOT_PASSWORD")
database_name = os.getenv("MONGODB_DB_NAME")
mongo_uri = f"mongodb://{adm_username}:{adm_password}@localhost:27017"

# Dicionários de níveis
niveis_idioma = {
    nivel: num
    for num, nivel in enumerate(
        [
            "Nenhum",
            "Básico",
            "Intermediário",
            "Avançado",
            "Fluente",
        ]
    )
}

niveis_ensino = {
    nivel: num
    for num, nivel in enumerate(
        [
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
        ]
    )
}

niveis_profissional = {
    nivel: num
    for num, nivel in enumerate(
        [
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
        ]
    )
}

# Tamanhos dos lotes (ajuste conforme necessário com base na memória/performance)
VAGA_BATCH_SIZE = 100
APPLICANT_BATCH_SIZE = 1_000

# =============================================================================
# CLASSES
# =============================================================================


# PyObjectId pode ser removida se não usada fora de Beanie
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
        name = "analise_requisitos"
        # use_state_management = True # Desabilitado por padrão, habilite se necessário

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
    pcd: Optional[str] = None
    nivel_academico: Optional[str] = None
    nivel_ingles: Optional[str] = None
    nivel_espanhol: Optional[str] = None


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================


def get_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    """Extrai o ID (_id ou id) como string de um documento."""
    doc_id = doc.get("_id")
    if doc_id is None:
        doc_id = doc.get("id")  # Fallback para campo 'id'

    if isinstance(doc_id, ObjectId):
        return str(doc_id)
    elif isinstance(doc_id, str):
        return doc_id
    elif doc_id is not None:
        # Tenta converter outros tipos, mas pode falhar
        try:
            return str(doc_id)
        except Exception:
            print(f"Aviso: Não foi possível converter ID '{doc_id}' para string.")
            return None
    return None


async def get_batch_from_cursor(
    cursor: AsyncIOMotorCursor, batch_size: int
) -> List[Dict[str, Any]]:
    """Lê um lote de documentos de um cursor motor."""
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
        logger.exception(f"Erro durante a iteração do cursor: {e}")
        # Dependendo do caso, você pode querer retornar o batch parcial
        # ou levantar a exceção novamente: raise e

    # print(f"Batch retornado! Tamanho: {len(batch)}")  # Print mais informativo
    return batch


async def find_existing_analysis_pairs(
    db: AsyncIOMotorDatabase, vaga_ids: List[str], candidato_ids: List[str]
) -> Set[Tuple[str, str]]:
    """Busca pares (id_vaga, id_candidato) existentes na coleção de análise."""
    if not vaga_ids or not candidato_ids:
        return set()

    collection_analise = db["analise_requisitos"]  # Usando motor diretamente
    existing_pairs = set()
    query = {"id_vaga": {"$in": vaga_ids}, "id_candidato": {"$in": candidato_ids}}
    projection = {
        "id_vaga": 1,
        "id_candidato": 1,
        "_id": 0,
    }  # Apenas campos necessários

    cursor = collection_analise.find(query, projection)
    async for doc in cursor:
        v_id = doc.get("id_vaga")
        c_id = doc.get("id_candidato")
        if v_id and c_id:
            existing_pairs.add((v_id, c_id))
    return existing_pairs


# =============================================================================
# FUNÇÕES DE PROCESSAMENTO (geradores e desclassificador mantidos, com ajustes)
# =============================================================================
def gerador_requisitos_para_vaga(vaga: dict) -> Requisitos:
    perfil = vaga.get("perfil_vaga", {})
    return Requisitos(
        pcd=perfil.get("vaga_especifica_para_pcd"),
        nivel_academico=perfil.get("nivel_academico"),
        nivel_ingles=perfil.get("nivel_ingles"),
        nivel_espanhol=perfil.get("nivel_espanhol"),
    )


def gerador_requisitos_para_candidato(candidato: dict) -> Requisitos:
    info_pessoais = candidato.get("informacoes_pessoais", {})
    formacao = candidato.get("formacao_e_idiomas", {})
    return Requisitos(
        pcd=info_pessoais.get("pcd"),
        nivel_academico=formacao.get("nivel_academico"),
        nivel_ingles=formacao.get("nivel_ingles"),
        nivel_espanhol=formacao.get("nivel_espanhol"),
    )


def desclassificador(
    requisitos_vaga: Requisitos, requisitos_candidato: Requisitos
) -> Tuple[bool, str, str]:
    # (Implementação mantida da resposta anterior, com verificações None)
    observacoes = []
    motivos = []
    desclassificar = False

    # 1. PCD
    if requisitos_vaga.pcd == "Sim":
        if requisitos_candidato.pcd == "Não":
            desclassificar = True
            motivos.append("Vaga para PCD.")
        elif not requisitos_candidato.pcd:
            observacoes.append("Candidato não possui dados sobre PCD.")

    # 2. Nível acadêmico
    req_acad_vaga_num = niveis_ensino.get(requisitos_vaga.nivel_academico)
    req_acad_cand_num = niveis_ensino.get(requisitos_candidato.nivel_academico)

    if req_acad_vaga_num is not None:
        if req_acad_cand_num is not None:
            if req_acad_vaga_num > req_acad_cand_num:
                desclassificar = True
                motivos.append(
                    f"Vaga requer '{requisitos_vaga.nivel_academico}', candidato tem '{requisitos_candidato.nivel_academico}'."
                )
        else:
            observacoes.append("Candidato não possui dados sobre formação acadêmica.")

    # 3. Nível idiomas
    def check_idioma(req_vaga, req_cand, idioma_nome):
        nonlocal desclassificar
        req_idioma_vaga_num = niveis_idioma.get(req_vaga)
        req_idioma_cand_num = niveis_idioma.get(req_cand)

        if req_idioma_vaga_num is not None and req_idioma_vaga_num > 0:
            if req_idioma_cand_num is not None:
                if req_idioma_vaga_num > req_idioma_cand_num:
                    desclassificar = True
                    motivos.append(
                        f"Vaga requer {idioma_nome} '{req_vaga.lower()}', candidato tem '{req_cand.lower()}'."
                    )
            else:
                observacoes.append(
                    f"Candidato não possui dados sobre nível de {idioma_nome}."
                )

    check_idioma(
        requisitos_vaga.nivel_ingles, requisitos_candidato.nivel_ingles, "inglês"
    )
    check_idioma(
        requisitos_vaga.nivel_espanhol, requisitos_candidato.nivel_espanhol, "espanhol"
    )

    return desclassificar, " ".join(motivos), " ".join(observacoes)


async def process_candidate_for_vaga(
    vaga: dict, candidato: dict, requisitos_vaga: Requisitos
) -> Optional[DocumentAnaliseRequisito]:
    """Processa um único candidato para uma vaga (CPU-bound + objeto)."""
    try:
        requisitos_candidato = gerador_requisitos_para_candidato(candidato)
        desclassificado, motivos, observacoes = desclassificador(
            requisitos_vaga, requisitos_candidato
        )

        vaga_id_str = get_doc_id(vaga)
        candidato_id_str = get_doc_id(candidato)

        if not vaga_id_str or not candidato_id_str:
            # Log já acontece em get_doc_id se falhar
            return None

        doc = DocumentAnaliseRequisito(
            id_vaga=vaga_id_str,
            id_candidato=candidato_id_str,
            desclassificado=desclassificado,
            motivos=motivos,
            observacoes=observacoes,
        )
        return doc
    except Exception as e:
        print(
            f"Erro processando candidato {get_doc_id(candidato)} para vaga {get_doc_id(vaga)}: {e}"
        )
        return None


def eta(batch_size: int, total_items: int, start_time: float) -> str:
    """Calcula o tempo estimado de chegada (ETA) para o processamento de lotes."""
    elapsed_time = time() - start_time
    eta_seconds = (elapsed_time / batch_size) * (total_items - batch_size)
    eta = time() + eta_seconds
    return f"{time.strftime('%H:%M:%S', time.localtime(eta))}"


# =============================================================================
# FUNÇÕES ASYNC PRINCIPAIS
# =============================================================================


async def init() -> AsyncIOMotorDatabase:
    """Inicializa Beanie e retorna o objeto de banco de dados motor."""
    client = AsyncIOMotorClient(mongo_uri)
    db = client[database_name]
    await init_beanie(database=db, document_models=[DocumentAnaliseRequisito])
    print("Conexão com Beanie inicializada.")
    return db


async def populate_analises_batch(db: AsyncIOMotorDatabase):
    """Popula análises de forma otimizada usando lotes e concorrência."""
    # Coleção de vagas
    collection_vagas = db["vagas"]
    estimated_count_vagas = await collection_vagas.estimated_document_count()

    # Coleção de candidatos
    collection_applicants = db["applicants"]
    estimated_count_candidatos = await collection_applicants.estimated_document_count()

    total_vagas_processadas = 0
    total_analises_criadas = 0

    vagas_cursor = collection_vagas.find({})  # Cursor para vagas
    logger.info("Iniciando processamento em lotes...")

    t0_vagas = time()
    while True:  # Loop para lotes de vagas
        logger.debug(f"\nBuscando lote de vagas (até {VAGA_BATCH_SIZE})...")
        vagas_batch = await get_batch_from_cursor(vagas_cursor, VAGA_BATCH_SIZE)
        if not vagas_batch:
            logger.info("Nenhuma vaga restante para processar.")
            break  # Fim das vagas

        vaga_ids_batch = [
            v_id for v_id in (get_doc_id(vaga) for vaga in vagas_batch) if v_id
        ]
        if not vaga_ids_batch:
            logger.info("Lote de vagas sem IDs válidos, pulando.")
            continue

        total_vagas_processadas += len(vagas_batch)
        logger.debug(
            f"Vagas: processando {total_vagas_processadas} de {estimated_count_vagas} (ETA: {eta(VAGA_BATCH_SIZE, estimated_count_vagas, t0_vagas)})"
        )

        # Mapeia ID da vaga para o objeto Requisitos da vaga (evita recalcular)
        requisitos_vaga_map = {
            get_doc_id(v): gerador_requisitos_para_vaga(v)
            for v in vagas_batch
            if get_doc_id(v)
        }

        applicants_cursor = collection_applicants.find({})

        total_candidatos_neste_lote_vaga = 0
        analises_criadas_neste_lote_vaga = 0

        t0_candidatos = time()
        while True:  # Loop para lotes de candidatos
            applicants_batch = await get_batch_from_cursor(
                applicants_cursor, APPLICANT_BATCH_SIZE
            )

            if not applicants_batch:
                logger.debug("Nenhum candidato restante para este lote de vagas.")
                break

            applicant_ids_batch = [
                a_id for a_id in (get_doc_id(app) for app in applicants_batch) if a_id
            ]

            if not applicant_ids_batch:
                logger.debug("Lote de candidatos sem IDs válidos, pulando.")
                continue

            total_candidatos_neste_lote_vaga += len(applicants_batch)
            logger.debug(
                f"Candidatos: processando {total_candidatos_neste_lote_vaga} de {estimated_count_candidatos} (ETA: {eta(APPLICANT_BATCH_SIZE, estimated_count_candidatos, t0_candidatos)})"
            )

            # 1. Otimização: Buscar pares existentes *apenas para os lotes atuais*
            # print("  Buscando análises existentes para os lotes atuais...")
            existing_pairs = await find_existing_analysis_pairs(
                db, vaga_ids_batch, applicant_ids_batch
            )
            # print(f"  Encontradas {len(existing_pairs)} análises existentes neste cruzamento de lotes.")

            # 2. Processamento (Concorrente ou Sequencial dentro do lote)
            docs_para_inserir_neste_sub_lote = []
            # Para processamento concorrente (opcional, pode consumir mais CPU)
            tasks = []

            for vaga in vagas_batch:
                vaga_id = get_doc_id(vaga)
                if not vaga_id:
                    continue
                requisitos_vaga = requisitos_vaga_map.get(vaga_id)
                if not requisitos_vaga:
                    continue  # Deveria existir, mas por segurança

                for candidato in applicants_batch:
                    candidato_id = get_doc_id(candidato)
                    if not candidato_id:
                        continue

                    # Processa apenas se o par (vaga, candidato) não existe
                    if (vaga_id, candidato_id) not in existing_pairs:
                        # Opção 1: Sequencial (menos CPU, pode ser mais lento se desclassificador for pesado)
                        # doc = await process_candidate_for_vaga(vaga, candidato, requisitos_vaga)
                        # if doc:
                        #     docs_para_inserir_neste_sub_lote.append(doc)

                        # Opção 2: Concorrente (mais CPU, potencialmente mais rápido se I/O for gargalo)
                        # Descomente a linha abaixo e comente a Opção 1 para usar concorrência
                        task = asyncio.create_task(
                            process_candidate_for_vaga(vaga, candidato, requisitos_vaga)
                        )
                        tasks.append(task)

            if tasks:  # Se usou a Opção 2 (concorrente)
                # print(f"  Aguardando {len(tasks)} tarefas de processamento...")
                results = await asyncio.gather(*tasks)
                docs_para_inserir_neste_sub_lote = [
                    doc for doc in results if doc is not None
                ]

            # 3. Bulk Insert para o cruzamento dos lotes atuais
            if docs_para_inserir_neste_sub_lote:
                # print(f"  Inserindo {len(docs_para_inserir_neste_sub_lote)} novas análises em lote...")
                try:
                    # Usando Beanie para insert_many (poderia usar motor direto também)
                    await DocumentAnaliseRequisito.insert_many(
                        docs_para_inserir_neste_sub_lote,
                        ordered=False,  # Continua inserindo mesmo se houver erro em um doc
                    )
                    analises_criadas_neste_lote_vaga += len(
                        docs_para_inserir_neste_sub_lote
                    )
                    # print("  Lote inserido com sucesso.")
                except Exception as e:
                    logger.exception(f"Erro ao inserir lote de análises: {e}")
            # else:
            # print("  Nenhum documento novo para inserir neste sub-lote.")

        logger.debug(f"Fim do processamento de candidatos para o lote de vagas atual.")
        logger.debug(
            f"  Total de candidatos verificados neste lote de vagas: {total_candidatos_neste_lote_vaga}"
        )
        logger.debug(
            f"  Novas análises criadas neste lote de vagas: {analises_criadas_neste_lote_vaga}"
        )
        total_analises_criadas += analises_criadas_neste_lote_vaga

    logger.debug("\n-----------------------------------------")
    logger.info("Processamento de todos os lotes concluído.")
    logger.info(f"Total de vagas processadas: {total_vagas_processadas}")
    logger.info(f"Total de análises criadas: {total_analises_criadas}")
    logger.debug("-----------------------------------------")


# =============================================================================
# MAIN
# =============================================================================
async def main():
    db = await init()  # Obtem o objeto db do motor
    await populate_analises_batch(db)


if __name__ == "__main__":
    print("Iniciando script de análise em lote...")
    start_time = datetime.now()
    # Adicionar criação de índice se necessário (executar apenas uma vez ou verificar existência)
    # Exemplo:
    # db_instance = await init()
    # await db_instance["analise_requisitos"].create_index([("id_vaga", 1), ("id_candidato", 1)], name="vaga_candidato_idx", unique=True) # unique=True se o par deve ser único
    # print("Índice vaga_candidato_idx garantido.")
    # await populate_analises_batch(db_instance)
    asyncio.run(main())  # Executa a lógica principal
    end_time = datetime.now()
    print(f"Script finalizado. Tempo total: {end_time - start_time}")
