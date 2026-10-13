"""Genera un conjunto sintético reproducible para demo de expedientes de terrazas.

No son expedientes ni estadísticas reales del Ayuntamiento de Madrid.
"""
import csv
import random
from pathlib import Path

# Semilla fija para que el CSV pueda regenerarse y compararse en una práctica.
# La repetibilidad no convierte los valores inventados en datos municipales reales.
SEED = 4200513
N_CASES = 1000
DISTRICTS = [
    "Centro", "Arganzuela", "Retiro", "Salamanca", "Chamartín", "Tetuán",
    "Chamberí", "Fuencarral-El Pardo", "Moncloa-Aravaca", "Latina",
    "Carabanchel", "Usera", "Puente de Vallecas", "Moratalaz", "Ciudad Lineal",
    "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicálvaro",
    "San Blas-Canillejas", "Barajas",
]
PROCEDURES = ["alta", "renovación", "modificación"]
FIELDS = [
    "expediente_id", "distrito", "tipo_tramite", "superficie_solicitada_m2",
    "requerimientos_documentacion", "mes_presentacion", "x", "y",
]


def main():
    rng = random.Random(SEED)
    rows = []
    for i in range(1, N_CASES + 1):
        district = rng.choice(DISTRICTS)
        # Las proporciones solo dan variedad al ejemplo; no son frecuencias reales
        # de expedientes de Madrid ni deben usarse como estimación estadística.
        procedure = rng.choices(PROCEDURES, weights=(0.42, 0.36, 0.22), k=1)[0]
        area = round(rng.uniform(8, 120), 1)
        requirements = rng.choices([0, 1, 2, 3], weights=(0.48, 0.32, 0.15, 0.05), k=1)[0]
        month = rng.randint(1, 12)
        procedure_weight = {"alta": 15, "renovación": 3, "modificación": 9}[procedure]
        # Índice artificial que combina campos del caso para disponer de una x
        # numérica. No representa un baremo oficial ni una regla administrativa.
        complexity = round(min(100, max(5, 18 + area * 0.32 + requirements * 10 + procedure_weight + rng.gauss(0, 7))), 2)
        # Relación cuadrática inventada para que el conjunto sirva como ejercicio de regresión.
        days = round(max(8, 18 + 0.34 * complexity + 0.009 * complexity**2 + rng.gauss(0, 7)))
        rows.append({
            "expediente_id": f"DEMO-TERR-2026-{i:04d}",
            "distrito": district,
            "tipo_tramite": procedure,
            "superficie_solicitada_m2": f"{area:.1f}",
            "requerimientos_documentacion": requirements,
            "mes_presentacion": month,
            "x": f"{complexity:.2f}",
            "y": days,
        })

    out = Path(__file__).resolve().parents[1] / "ejemplos" / "datos_ejemplo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Escritos {len(rows)} expedientes sintéticos en {out}")


if __name__ == "__main__":
    main()
