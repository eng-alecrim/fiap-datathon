# =============================================================
# LIBRARIES
# =============================================================

from abc import ABC, abstractmethod

import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from typing import List, Union, Type

# =============================================================
# Functions
# =============================================================


def create_word2vec(
    texts: List[Union[str, List[str]]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 0,
) -> Word2Vec:
    """
    Cria um modelo Word2Vec do gensim a partir de uma lista de textos.

    Parâmetros:
        texts (list): Lista de strings ou listas tokenizadas.
        vector_size (int): Dimensão dos vetores de palavras.
        window (int): Distância máxima entre a palavra atual e a prevista.
        min_count (int): Ignora palavras com frequência total inferior a este valor.
        workers (int): Número de threads para treinamento.
        sg (int): Algoritmo de treinamento: 1 para skip-gram; caso contrário, CBOW.

    Retorna:
        model: Modelo Word2Vec do gensim treinado.
    """

    # Tokenize texts: if an element is a string, split by whitespace.
    tokenized_texts = [
        text.split() if isinstance(text, str) else text for text in texts
    ]

    # Create and train the model
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )
    return model


# =============================================================
# Strategy Pattern
# =============================================================

# -------------------------------------------------------------
# Interface
# -------------------------------------------------------------


class TextEmbeddingStrategy(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> np.ndarray:
        pass


# -------------------------------------------------------------
# Concrete Strategies
# -------------------------------------------------------------


class Word2VecEmbeddingStrategy(TextEmbeddingStrategy):
    def __init__(self, model: Word2Vec) -> None:
        self.model = model

    def create_embedding(self, text: str) -> np.ndarray:
        tokens = text.split()
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)


# Concrete strategy for SentenceTransformer
class SentenceTransformerEmbeddingStrategy(TextEmbeddingStrategy):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        precision: str = "float32",
        device: str = "cpu",
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.precision = precision

    def create_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(sentences=text, precision=self.precision)


# -------------------------------------------------------------
# Context
# -------------------------------------------------------------


class EmbeddingModel:
    def __init__(
        self, strategy: Type[TextEmbeddingStrategy] | None = None, **kwargs
    ) -> None:
        strategy_instance = strategy(**kwargs) if strategy else None
        if not isinstance(strategy_instance, TextEmbeddingStrategy):
            raise ValueError("strategy must be an instance of TextEmbeddingStrategy")
        self.strategy = strategy_instance

    def set_strategy(self, strategy: Type[TextEmbeddingStrategy], **kwargs) -> None:
        """
        Define a estratégia de embedding a ser utilizada.

        Parâmetros:
            strategy (TextEmbeddingStrategy): Estratégia de embedding.
        """
        strategy_instance = strategy(**kwargs)
        if not isinstance(strategy_instance, TextEmbeddingStrategy):
            raise ValueError("strategy must be an instance of TextEmbeddingStrategy")
        self.strategy = strategy_instance

    def create_embedding(self, text: str) -> np.ndarray:
        if not isinstance(self.strategy, TextEmbeddingStrategy):
            raise ValueError("strategy is not an instance of TextEmbeddingStrategy")
        return self.strategy.create_embedding(text)
