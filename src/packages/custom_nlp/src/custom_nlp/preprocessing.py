# =============================================================
# LIBRARIES
# =============================================================

import unicodedata
from abc import ABC, abstractmethod

import nltk
import spacy
from nltk.stem import RSLPStemmer

# =============================================================
# Constants
# =============================================================

nlp_pt = spacy.load("pt_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")

# =============================================================
# Strategy Pattern
# =============================================================

# -------------------------------------------------------------
# Interface
# -------------------------------------------------------------


class TextProcessingStrategy(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass


# -------------------------------------------------------------
# Concrete Strategies
# -------------------------------------------------------------


class NormalizationStrategy(TextProcessingStrategy):
    def process(self, text: str) -> str:
        """
        Processa e normaliza o texto removendo acentos, tags HTML e caracteres especiais.

        Parâmetros:
            text (str): Texto a ser normalizado.

        Retorna:
            str: Texto normalizado.
        """
        import re

        text = str(text)
        nfkd_form = unicodedata.normalize("NFC", text)
        output_str = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
        regex_tags = r"</?.>"
        output_str = re.sub(regex_tags, "", output_str)
        regex = re.compile(r"[^a-zA-Z_À-ÿ\s]+")
        tokens = regex.sub(" ", output_str).split()
        tokens = list(map(lambda x: x.lower(), tokens))
        return " ".join(map(lambda x: x.strip(), tokens))


# Concrete strategy for stopwords removal
class StopwordsRemovalStrategy(TextProcessingStrategy):
    def __init__(self, language: str = "portuguese") -> None:
        """
        Inicializa a estratégia de remoção de stopwords.
        Parâmetros:
            language (str): Idioma para a remoção de stopwords. Padrão é "portuguese".
        """
        super().__init__()
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        self.language = language

    def process(self, text: str) -> str:
        """
        Remove as stopwords do texto utilizando NLTK e palavras do Spacy.

        Parâmetros:
            text (str): Texto do qual as stopwords serão removidas.

        Retorna:
            str: Texto sem as stopwords.
        """
        tokens = nltk.word_tokenize(text, language=self.language)
        stopwords = set(nltk.corpus.stopwords.words(self.language))
        (
            stopwords.update(spacy.lang.pt.stop_words.STOP_WORDS)
            if self.language == "portuguese"
            else stopwords.update(spacy.lang.en.stop_words.STOP_WORDS)
        )
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
        return " ".join(filtered_tokens)


# Concrete strategy for stemming
class StemmingStrategy(TextProcessingStrategy):
    def process(self, text: str) -> str:
        """
        Aplica stemming ao texto utilizando o RSLPStemmer do NLTK.

        Parâmetros:
            text (str): Texto a ser submetido ao stemming.

        Retorna:
            str: Texto com palavras reduzidas à sua raiz.
        """
        tokens = nltk.word_tokenize(text, language="portuguese")
        stemmer = RSLPStemmer()
        stemmed = [stemmer.stem(token) for token in tokens]
        return " ".join(stemmed)


# Concrete strategy for lemmatization
class LemmatizationStrategy(TextProcessingStrategy):
    def process(self, text: str) -> str:
        """
        Aplica lemmatização ao texto utilizando o modelo Spacy.

        Parâmetros:
            text (str): Texto a ser lematizado.

        Retorna:
            str: Texto lematizado.
        """
        doc = nlp_pt(text)
        lemmatized = [token.lemma_ for token in doc]
        return " ".join(lemmatized)


# -------------------------------------------------------------
# Context
# -------------------------------------------------------------


# Update the Preprocessor to use a strategy object
class Preprocessor:
    """
    Classe que aplica estratégias de processamento de texto.
    """

    def __init__(self, strategy: TextProcessingStrategy = None) -> None:
        """
        Inicializa o Preprocessor com uma estratégia opcional.

        Parâmetros:
            strategy (TextProcessingStrategy, opcional): Estratégia de processamento a ser usada.
        """
        self.strategy = strategy

    def set_strategy(self, strategy: TextProcessingStrategy) -> None:
        """
        Define a estratégia de processamento a ser utilizada.

        Parâmetros:
            strategy (TextProcessingStrategy): Estratégia a ser aplicada.
        """
        self.strategy = strategy

    def apply(self, text: str) -> str:
        """
        Aplica a estratégia de processamento configurada ao texto.

        Parâmetros:
            text (str): Texto a ser processado.

        Retorna:
            str: Texto processado.
        """
        return self.strategy.process(text)
