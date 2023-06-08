from abc import ABC, abstractmethod
class Dataset(ABC):

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def create():
        pass


