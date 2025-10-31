import random

from Search_Algorithms.Problem.Jogo8.Complexidade import Complexidade
from Search_Algorithms.Problem.Jogo8.Acao import Acao
from Search_Algorithms.Problem.Jogo8.Estado import Estado


class Jogo8:
    def __init__(self, tamanho: int, complexidade: Complexidade):
        self.estado_final: Estado = Estado(tamanho)
        self.complexidade: Complexidade = complexidade
        self.estado_inicial: Estado = self.criar_jogo()
        self.estado_atual: Estado = self.estado_inicial

    @staticmethod
    def transicao(estado: Estado, acao: Acao) -> Estado:
        aux = 0
        if acao == Acao.Cima:
            if estado.vazio in range(estado.tamanho):
                return estado
            else:
                aux = estado.vazio - estado.tamanho
        elif acao == Acao.Baixo:
            if estado.vazio in range(estado.tamanho ** 2 - 1, estado.tamanho ** 2 - 1 - estado.tamanho, -1):
                return estado
            else:
                aux = estado.vazio + estado.tamanho
        elif acao == Acao.Direita:
            if estado.vazio in range(estado.tamanho - 1, estado.tamanho ** 2, estado.tamanho):
                return estado
            else:
                aux = estado.vazio + 1
        elif acao == Acao.Esquerda:
            if estado.vazio in range(0, estado.tamanho ** 2, estado.tamanho):
                return estado
            else:
                aux = estado.vazio - 1
        novo_tabuleiro = list(estado.tabuleiro)
        novo_tabuleiro[estado.vazio] = novo_tabuleiro[aux]
        novo_tabuleiro[aux] = 0
        return Estado(estado.tamanho, tuple(novo_tabuleiro), aux)

    def criar_jogo(self) -> Estado:
        novo_estado = self.estado_final
        for _ in range(novo_estado.tamanho ** self.complexidade.value):
            acao = random.randrange(len(Acao))
            novo_estado = Jogo8.transicao(novo_estado, Acao(acao))
        return novo_estado
