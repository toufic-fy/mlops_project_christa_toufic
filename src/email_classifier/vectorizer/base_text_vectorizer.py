from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseTextVectorizer(ABC):
    @abstractmethod
    def vectorize(self, documents: List) -> List:
        pass