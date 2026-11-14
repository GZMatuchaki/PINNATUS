from physics.geometry import Rectangle
from utils.plots import plot_points

canal = Rectangle(L=1.0, H=1.0, x_origem=-2.5, y_origem=-0.5)     # use os nomes dos seus campos
dominio = canal.sample_domain(100, metodo="lhs")
borda = canal.sample_boundary(40, metodo="lhs") # pontos são alocados tendo o centro como referencia

plot_points(canal, dominio, borda,
            titulo="Canal: pontos de treino",
            salvar="outputs/pontos_treino.png")