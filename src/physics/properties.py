"""Propriedades do fluido (em unidades do SI)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FluidoNewtoniano:
    rho: float   # massa específica [kg/m³]
    mu: float    # viscosidade dinâmica [Pa·s]

    @property
    def nu(self) -> float:
        """Viscosidade cinemática [m²/s]."""
        return self.mu / self.rho


# %%  Teste rápido
if __name__ == "__main__":
    agua_mestrado = FluidoNewtoniano(rho=1.0, mu=1.0)
    print(agua_mestrado)
    print("nu =", agua_mestrado.nu, "m²/s")