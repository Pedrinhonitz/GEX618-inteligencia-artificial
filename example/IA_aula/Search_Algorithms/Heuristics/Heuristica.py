from abc import abstractmethod

from Search_Algorithms.Problem.Jogo8.Estado import Estado


class Heuristic:
    def __init__(self, nome: str, objetivo: Estado):
        self.nome: str = nome
        self.objetivo: Estado = objetivo

    @abstractmethod
    def computar(self, estado: Estado) -> int:
        pass


class Heuristic1(Heuristic):

    def __init__(self, objetivo: Estado):
        super().__init__("Quantidade de peças fora do lugar", objetivo)

    def computar(self, estado: Estado) -> int:
        h = 0
        for idx, elemento in enumerate(estado.tabuleiro):
            if elemento != 0 and self.objetivo.tabuleiro[idx] != elemento:
                h += 1
        return h


class Heuristic2(Heuristic):

    def __init__(self, objetivo: Estado):
        super().__init__("Distância de quarteirão", objetivo)

    def computar(self, estado: Estado) -> int:
        h = 0
        for idx, elemento in enumerate(estado.tabuleiro):
            if elemento != 0 and self.objetivo.tabuleiro[idx] != elemento:
                idx_objetivo = abs(elemento - 1)
                h += abs(estado.retornar_coluna_peca(idx) - self.objetivo.retornar_coluna_peca(idx_objetivo))
                h += abs(estado.retornar_linha_peca(idx) - self.objetivo.retornar_linha_peca(idx_objetivo))
        return h
