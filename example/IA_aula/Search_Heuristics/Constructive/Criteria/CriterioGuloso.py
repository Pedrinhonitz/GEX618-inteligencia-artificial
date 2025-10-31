from abc import abstractmethod


class CriterioGuloso:
    def __init__(self, nome: str, distancias: tuple):
        self.nome: str = nome
        self.distancias: tuple = distancias
        self.tamanho: int = len(distancias)

    @abstractmethod
    def melhor_elemento(self, nao_visitados: set[int], ultimo: int) -> int:
        pass

    @abstractmethod
    def melhor_alfa(self, nao_visitados: set[int], ultimo: int, alfa: float) -> int:
        pass
