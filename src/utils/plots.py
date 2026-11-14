###==========Plotagem da geometria e amostragem==========###

"""Gráficos de pontos de treino (domínio e contorno)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as RetanguloMPL

# Cores e tinta (paleta validada: azul x vermelho passa em daltonismo e contraste)
COR_DOMINIO = "#2a78d6"   # azul
COR_CONTORNO = "#e34948"  # vermelho
TINTA = "#0b0b0b"         # texto principal
TINTA_2 = "#52514e"       # texto secundário
TINTA_FRACA = "#898781"   # eixos e rótulos
LINHA_EIXO = "#c3c2b7"    # contorno do retângulo e molduras
FUNDO = "#fcfcfb"         # fundo do gráfico


def plot_points(
    geom,
    domain: np.ndarray,
    boundary: dict,
    titulo: str = "Pontos de treino",
    zoom: float = 0.06,
    salvar: str | None = None,
    mostrar: bool = True,
):
    """Desenha os pontos do domínio (azul) e do contorno (vermelho).

    geom     : objeto com x_min, x_max, y_min, y_max (ex.: Rectangle)
    domain   : array (n, 2) com os pontos do domínio
    boundary : dicionário {"bottom": array, "right": ..., "top": ..., "left": ...}
    zoom     : lado [m] da janela de zoom nos dois cantos da entrada
    salvar   : caminho do arquivo (.png/.pdf) ou None
    """
    contorno = np.vstack(list(boundary.values()))
    L = geom.x_max - geom.x_min
    H = geom.y_max - geom.y_min

    # 1. Figura: canal inteiro em cima, dois zooms (cantos da entrada) embaixo
    fig = plt.figure(figsize=(12, 6.2), facecolor=FUNDO)
    grade = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.6, wspace=0.18)
    ax = fig.add_subplot(grade[0, :])
    ax_inf = fig.add_subplot(grade[1, 0])
    ax_sup = fig.add_subplot(grade[1, 1])

    # 2. Painel principal
    _desenhar(ax, geom, domain, contorno, s_dom=0.6, s_cont=1.2)
    folga = 0.04 * H
    ax.set_xlim(geom.x_min - folga, geom.x_max + folga)
    ax.set_ylim(geom.y_min - folga, geom.y_max + folga)
    ax.set_aspect("equal")
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=13, pad=30)
    contagens = " · ".join(f"{k} {len(v)}" for k, v in boundary.items())
    ax.text(0.0, 1.04, f"{L:g} m × {H:g} m   |   contorno por aresta: {contagens}",
            transform=ax.transAxes, color=TINTA_2, fontsize=9.5, va="bottom")

    # Legenda: na linha do título, à direita
    ax.scatter([], [], s=30, color=COR_DOMINIO, label=f"domínio ({len(domain)})")
    ax.scatter([], [], s=30, color=COR_CONTORNO, label=f"contorno ({len(contorno)})")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.13), ncol=2, frameon=False,
              fontsize=10, labelcolor=TINTA_2, handletextpad=0.2, columnspacing=1.4,
              borderaxespad=0)

    # 3. Zooms nos dois cantos da entrada (onde as condições se encontram)
    cantos = {
        ax_inf: (geom.x_min, geom.y_min, "canto inferior"),
        ax_sup: (geom.x_min, geom.y_max, "canto superior"),
    }
    for eixo, (xc, yc, nome) in cantos.items():
        _desenhar(eixo, geom, domain, contorno, s_dom=9, s_cont=16)
        x0 = xc - 0.15 * zoom
        y0 = yc - 0.15 * zoom if yc == geom.y_min else yc - 0.85 * zoom
        eixo.set_xlim(x0, x0 + zoom)
        eixo.set_ylim(y0, y0 + zoom)
        eixo.set_aspect("equal")
        # Distância do ponto de contorno mais próximo até o canto
        d = np.hypot(contorno[:, 0] - xc, contorno[:, 1] - yc).min()
        eixo.set_title(f"zoom: {nome}\nponto de contorno mais próximo do canto: {d * 1000:.2f} mm",
                       loc="left", color=TINTA_2, fontsize=9.5, linespacing=1.5)
        eixo.plot(xc, yc, marker="x", ms=7, mew=1.4, color=TINTA, zorder=4)
        # Moldura do zoom no painel principal
        ax.add_patch(RetanguloMPL((x0, y0), zoom, zoom, fill=False,
                                  ec=TINTA_2, lw=0.9, zorder=5))

    fig.align_labels()
    if salvar:
        fig.savefig(salvar, dpi=200, bbox_inches="tight", facecolor=FUNDO)
    if mostrar:
        plt.show()
    return fig


def _desenhar(eixo, geom, domain, contorno, s_dom, s_cont):
    """Desenha um painel: borda do retângulo, domínio e contorno (uso interno)."""
    eixo.set_facecolor(FUNDO)
    eixo.add_patch(RetanguloMPL((geom.x_min, geom.y_min), geom.x_max - geom.x_min,
                                geom.y_max - geom.y_min, fill=False,
                                ec=LINHA_EIXO, lw=1.0, zorder=1))
    eixo.scatter(domain[:, 0], domain[:, 1], s=s_dom, color=COR_DOMINIO,
                 linewidths=0, alpha=0.75, zorder=2, rasterized=True)
    eixo.scatter(contorno[:, 0], contorno[:, 1], s=s_cont, color=COR_CONTORNO,
                 linewidths=0, zorder=3, rasterized=True)
    eixo.set_xlabel("x [m]", color=TINTA_FRACA, fontsize=9)
    eixo.set_ylabel("y [m]", color=TINTA_FRACA, fontsize=9)
    eixo.tick_params(colors=TINTA_FRACA, labelsize=8, length=3)
    for lado in eixo.spines.values():
        lado.set_visible(False)
