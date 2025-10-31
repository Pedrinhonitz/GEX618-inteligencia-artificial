from Search_Algorithms.Heuristics.Heuristica import Heuristic
from Search_Algorithms.Problem.Jogo8.Estado import Estado
from Search_Algorithms.Problem.Jogo8.Acao import Acao


class Arvore:
    def __init__(self, estado: Estado, nivel: int, heuristica: Heuristic | None = None, pai: 'Arvore' = None, acao: Acao = None, ):
        self.estado: Estado = estado
        self.filhos: list[Arvore] = []
        self.pai: Arvore = pai
        self.acao: Acao = acao
        self.nivel: int = nivel
        self.heuristica: Heuristic = heuristica
        self.valor_h: int = None if self.heuristica is None else heuristica.computar(self.estado)

    def __lt__(self, arvore: 'Arvore'):
        return self.valor_h + self.nivel < arvore.valor_h + arvore.nivel

    def adicionar_filho(self, estado: Estado, acao: Acao) -> 'Arvore':
        novo_filho = Arvore(estado, self.nivel + 1, self.heuristica, self, acao)
        self.filhos.append(novo_filho)
        return novo_filho

    def retornar_acoes(self) -> list[Acao]:
        acoes = []
        arvore = self
        while True:
            if arvore.acao is None:
                break
            acoes.append(arvore.acao)
            arvore = arvore.pai
        acoes.reverse()
        return acoes
