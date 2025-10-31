from Search_Heuristics.Solucao import Solucao
from abc import abstractmethod


class Vizinhanca:
    def __init__(self, nome: str, distancias: tuple[tuple[int]], qtd_trocas: int):
        self.nome: str = nome
        self.distancias: tuple[tuple[int]] = distancias
        self.qtd_trocas: int = qtd_trocas
        self.tamanho: int = len(distancias)

    @abstractmethod
    def computar_qualidade(self, solucao: Solucao, i: int, j: int) -> int:
        pass

    @staticmethod
    @abstractmethod
    def gerar_novo_ciclo(solucao: Solucao, i: int, j: int) -> list:
        pass

    @abstractmethod
    def melhor_vizinho(self, solucao: Solucao, tabu: set) -> Solucao:
        pass

    @abstractmethod
    def primeiro_vizinho_melhor(self, solucao: Solucao, tabu: set) -> Solucao:
        pass
