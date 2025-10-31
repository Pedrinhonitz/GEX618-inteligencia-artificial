class Estado:
    def __init__(self, tamanho: int, tabuleiro: tuple[int, ...] = (), vazio: int = -1):
        self.tamanho: int = tamanho
        self.tabuleiro: tuple[int, ...] = self.criar_tabuleiro() if len(tabuleiro) == 0 else tabuleiro
        self.vazio: int = self.tamanho ** 2 - 1 if vazio == -1 else vazio

    def __str__(self):
        tabuleiro_str = " " + "_____" * self.tamanho
        for i in range(self.tamanho):
            tabuleiro_str += "\n| "
            for j in range(self.tamanho):
                numero = self.tabuleiro[i * self.tamanho + j]
                espaco = ""
                if numero < 10:
                    espaco = " "
                tabuleiro_str += espaco + str(numero) + " | "
        tabuleiro_str += "\n " + "-----" * self.tamanho
        return tabuleiro_str

    def __eq__(self, estado: 'Estado') -> bool:
        if self.tamanho != estado.tamanho:
            return False
        for i in range(len(self.tabuleiro)):
            if self.tabuleiro[i] != estado.tabuleiro[i]:
                return False
        return True

    def __ne__(self, estado: 'Estado') -> bool:
        return not self.__eq__(estado)

    def criar_tabuleiro(self) -> tuple[int, ...]:
        tabuleiro = []
        for i in range(1, self.tamanho**2):
            tabuleiro.append(i)
        tabuleiro.append(0)
        return tuple(tabuleiro)

    def retornar_linha_peca(self, idx: int) -> int:
        return idx // self.tamanho

    def retornar_coluna_peca(self, idx: int) -> int:
        return idx % self.tamanho
