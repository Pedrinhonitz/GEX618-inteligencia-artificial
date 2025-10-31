from enum import Enum


class Acao(Enum):
    Cima = 0
    Baixo = 1
    Direita = 2
    Esquerda = 3

    def inversa(self) -> 'Acao':
        if self == Acao.Cima:
            return Acao.Baixo
        elif self == Acao.Baixo:
            return Acao.Cima
        elif self == Acao.Direita:
            return Acao.Esquerda
        else:
            return Acao.Direita
