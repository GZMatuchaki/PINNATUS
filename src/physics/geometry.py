"""Geometria do domínio retangular (em unidades do SI).

O retângulo descreve apenas a forma: dimensões, limites, normais e sorteio de
pontos no domínio e no contorno. Ele não sabe nada de condições de contorno;
quem decide o que é entrada, parede ou saída é o problema (problems/).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


# %% 1. Retângulo
@dataclass(frozen=True)
class Rectangle:
    L: float               # comprimento, na direção x [m]
    H: float               # altura, na direção y [m]
    x_origem: float = 0.0  # x do canto inferior esquerdo [m]
    y_origem: float = 0.0  # y do canto inferior esquerdo [m]

    # 1.1 Grandezas geométricas (só dependem dos atributos)
    @property
    def area(self) -> float:
        return self.L * self.H

    @property
    def perimetro(self) -> float:
        return 2 * (self.L + self.H)

    # 1.2 Limites das arestas
    @property
    def x_min(self) -> float:
        return self.x_origem

    @property
    def x_max(self) -> float:
        return self.x_origem + self.L

    @property
    def y_min(self) -> float:
        return self.y_origem

    @property
    def y_max(self) -> float:
        return self.y_origem + self.H

    # 1.3 Normais externas de cada aresta
    @property
    def normais(self) -> dict[str, np.ndarray]:
        return {
            "bottom": np.array([0.0, -1.0]),
            "right": np.array([1.0, 0.0]),
            "top": np.array([0.0, 1.0]),
            "left": np.array([-1.0, 0.0]),
        }

    # %% 2. Amostragem no domínio (interior)
    def sample_domain(
        self,
        n_samples: int,
        seed: int | None = 42,
        metodo: str = "lhs",
    ) -> np.ndarray:
        """Retorna um array (n_samples, 2) com pontos no interior do retângulo."""
        if n_samples <= 0:
            raise ValueError("n_samples deve ser positivo.")

        if metodo == "pseudo":
            return self.sample_domain_pseudo(n_samples, seed)
        elif metodo == "lhs":
            return self.sample_domain_lhs(n_samples, seed)
        elif metodo == "sobol":
            return self.sample_domain_sobol(n_samples, seed)
        elif metodo == "grid":
            return self.sample_domain_grid(n_samples)
        else:
            raise ValueError(
                f"Método de domínio '{metodo}' não reconhecido. "
                "Use 'pseudo', 'lhs', 'sobol' ou 'grid'."
            )

    # 2.1 Pseudoaleatório (uniforme)
    def sample_domain_pseudo(self, n_samples: int, seed: int | None = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x_samples = rng.uniform(self.x_min, self.x_max, n_samples)
        y_samples = rng.uniform(self.y_min, self.y_max, n_samples)
        return np.column_stack((x_samples, y_samples))

    # 2.2 Hipercubo latino
    def sample_domain_lhs(self, n_samples: int, seed: int | None = 42) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=2, rng=seed)
        pontos = sampler.random(n=n_samples)
        return qmc.scale(
            pontos,
            l_bounds=[self.x_min, self.y_min],
            u_bounds=[self.x_max, self.y_max],
        )

    # 2.3 Sobol (exige n_samples = 2^m)
    def sample_domain_sobol(self, n_samples: int, seed: int | None = 42) -> np.ndarray:
        if n_samples & (n_samples - 1) != 0:
            raise ValueError("Sobol exige n_samples = 2^m (ex.: 8192, 16384).")
        m = n_samples.bit_length() - 1
        sampler = qmc.Sobol(d=2, scramble=True, rng=seed)
        pontos = sampler.random_base2(m=m)
        return qmc.scale(
            pontos,
            l_bounds=[self.x_min, self.y_min],
            u_bounds=[self.x_max, self.y_max],
        )

    # 2.4 Grade regular (centros das células: nenhum ponto na fronteira)
    def sample_domain_grid(self, n_samples: int) -> np.ndarray:
        # Grade cartesiana completa: nx * ny deve ser exatamente n_samples.
        fatores = [
            (nx, n_samples // nx)
            for nx in range(1, n_samples + 1)
            if n_samples % nx == 0
        ]
        if n_samples > 1:
            fatores = [(nx, ny) for nx, ny in fatores if nx > 1 and ny > 1]
        if not fatores:
            raise ValueError("Para grid, use n_samples=1 ou um número composto.")

        # Escolhe a grade cujos espaçamentos em x e y mais se aproximam.
        nx, ny = min(
            fatores,
            key=lambda par: abs(np.log((self.L / par[0]) / (self.H / par[1]))),
        )
        x_samples = self.x_min + (np.arange(nx) + 0.5) * self.L / nx
        y_samples = self.y_min + (np.arange(ny) + 0.5) * self.H / ny
        grid_x, grid_y = np.meshgrid(x_samples, y_samples)
        return np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # %% 3. Amostragem no contorno
    # 3.1 Quantidade de pontos por aresta (método dos maiores restos)
    def boundary_counts(self, n_samples: int) -> dict[str, int]:
        """Distribui exatamente n_samples pontos pelo comprimento, com mínimo de 1 por aresta."""
        if n_samples < 4:
            raise ValueError("É necessário pelo menos 4 pontos de fronteira.")

        perimetro = self.perimetro
        ideal = {
            "left": n_samples * self.H / perimetro,
            "right": n_samples * self.H / perimetro,
            "top": n_samples * self.L / perimetro,
            "bottom": n_samples * self.L / perimetro,
        }

        counts = {lado: max(1, int(valor)) for lado, valor in ideal.items()}

        # Faltam pontos: dá 1 à aresta mais abaixo do ideal.
        while sum(counts.values()) < n_samples:
            lado = max(counts, key=lambda nome: ideal[nome] - counts[nome])
            counts[lado] += 1

        # Sobram pontos: tira 1 da aresta mais acima do ideal (mínimo de 1 por aresta).
        while sum(counts.values()) > n_samples:
            candidatos = [nome for nome in counts if counts[nome] > 1]
            lado = min(candidatos, key=lambda nome: ideal[nome] - counts[nome])
            counts[lado] -= 1

        if sum(counts.values()) != n_samples:
            raise RuntimeError("boundary_counts: a soma não fechou.")
        return counts

    # 3.2 Posições relativas t ao longo de uma aresta (uso interno)
    def _sample_1d(self, n: int, rng: np.random.Generator, metodo: str) -> np.ndarray:
        """Gera n posições relativas em (0, 1)."""
        if n <= 0:
            raise ValueError("n deve ser positivo.")
        if metodo == "grid":
            # Centros das células: t nunca é 0 nem 1, então nenhum ponto cai num canto.
            return (np.arange(n) + 0.5) / n
        elif metodo == "lhs":
            sampler = qmc.LatinHypercube(d=1, rng=rng)
            return sampler.random(n)[:, 0]
        else:
            raise ValueError(
                f"Método de fronteira '{metodo}' não reconhecido. Use 'grid' ou 'lhs'."
            )

    # 3.3 Amostragem das quatro arestas (separadas por nome)
    def sample_boundary(
        self,
        n_samples: int,
        seed: int | None = 42,
        metodo: str = "lhs",
    ) -> dict[str, np.ndarray]:
        """Retorna {aresta: pontos (n_aresta, 2)}, com n_samples pontos no total."""
        counts = self.boundary_counts(n_samples)
        rng = np.random.default_rng(seed)  # um gerador para as quatro arestas

        t = {lado: self._sample_1d(n, rng, metodo) for lado, n in counts.items()}

        x_bottom = self.x_min + t["bottom"] * self.L
        x_top = self.x_min + t["top"] * self.L
        y_left = self.y_min + t["left"] * self.H
        y_right = self.y_min + t["right"] * self.H

        return {
            "bottom": np.column_stack((x_bottom, np.full(counts["bottom"], self.y_min))),
            "right": np.column_stack((np.full(counts["right"], self.x_max), y_right)),
            "top": np.column_stack((x_top, np.full(counts["top"], self.y_max))),
            "left": np.column_stack((np.full(counts["left"], self.x_min), y_left)),
        }


# %% 4. Testes rápidos: executados apenas ao rodar este arquivo diretamente
if __name__ == "__main__":
    canal = Rectangle(L=5.0, H=1.0, x_origem=0.0, y_origem=0.0)
    canal_centrado = Rectangle(L=5.0, H=1.0, x_origem=-canal.L / 2, y_origem=-canal.H / 2)

    # 4.1 Grandezas e limites
    print("Área:", canal.area, "| Perímetro:", canal.perimetro)
    print("Canal (canto):", (canal.x_min, canal.x_max), (canal.y_min, canal.y_max))
    print("Canal centrado:",
          (canal_centrado.x_min, canal_centrado.x_max),
          (canal_centrado.y_min, canal_centrado.y_max))

    # 4.2 Domínio: formato, pontos dentro do retângulo e fora da fronteira
    print("\nDomínio (canal centrado):")
    for metodo, N in [("pseudo", 1000), ("lhs", 1000), ("sobol", 1024), ("grid", 1000)]:
        pontos = canal_centrado.sample_domain(n_samples=N, metodo=metodo)
        dentro = (
            (pontos[:, 0] > canal_centrado.x_min) & (pontos[:, 0] < canal_centrado.x_max)
            & (pontos[:, 1] > canal_centrado.y_min) & (pontos[:, 1] < canal_centrado.y_max)
        ).all()
        print(f"  {metodo:6s} {pontos.shape}  todos estritamente dentro: {dentro}")

    try:
        canal_centrado.sample_domain(n_samples=1000, metodo="sobol")
    except ValueError as erro:
        print("  sobol com 1000 pontos ->", erro)

    # 4.3 Contorno: contagens, normais, repetidos e distância aos cantos
    print("\nContagens por aresta:")
    for n in [4, 5, 6, 7, 4000, 4001]:
        c = canal.boundary_counts(n)
        print(f"  {n:5d} {c} soma ok: {sum(c.values()) == n}")

    cantos = np.array([
        [canal_centrado.x_min, canal_centrado.y_min],
        [canal_centrado.x_max, canal_centrado.y_min],
        [canal_centrado.x_max, canal_centrado.y_max],
        [canal_centrado.x_min, canal_centrado.y_max],
    ])
    print("\nContorno (canal centrado, 4000 pontos):")
    for metodo in ["grid", "lhs"]:
        borda = canal_centrado.sample_boundary(4000, metodo=metodo)
        todos = np.vstack(list(borda.values()))
        repetidos = len(todos) - len(np.unique(todos, axis=0))
        d_canto = np.linalg.norm(todos[:, None, :] - cantos[None, :, :], axis=2).min()
        print(f"  {metodo:4s} {todos.shape} repetidos: {repetidos} | "
              f"menor distância a um canto: {d_canto * 1000:.2f} mm")
        for lado, pts in borda.items():
            print(f"       {lado:6s} {pts.shape}  normal {canal_centrado.normais[lado]}")

    # 4.4 Reprodutibilidade
    a = canal.sample_boundary(4000)
    b = canal.sample_boundary(4000)
    print("\nReprodutível com a mesma semente:", all(np.array_equal(a[k], b[k]) for k in a))