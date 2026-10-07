import json

from config import TOP_K
from rag_system import RAGSystem

GOLDEN_SET_PATH = "golden_set.json"


def evaluar(rag: RAGSystem, golden_set: list[dict], modo: str) -> dict:
    detalle = []

    for caso in golden_set:
        docs = rag.buscar(caso["pregunta"], modo=modo)
        fuentes = [d.metadata["source"] for d in docs]
        esperado = caso["documento_id_esperado"]

        # Recall@k: ¿aparece el documento esperado entre los k recuperados? (0 o 1)
        recall = 1.0 if esperado in fuentes else 0.0

        # Precision@k: de los k recuperados, ¿qué proporción viene del documento esperado?
        precision = fuentes.count(esperado) / TOP_K

        detalle.append({
            "pregunta": caso["pregunta"],
            "esperado": esperado,
            "recuperados": fuentes,
            "recall": recall,
            "precision": precision,
        })

    n = len(detalle)
    return {
        "detalle": detalle,
        "recall": sum(d["recall"] for d in detalle) / n,
        "precision": sum(d["precision"] for d in detalle) / n,
    }


def imprimir_detalle(reporte: dict):
    for d in reporte["detalle"]:
        estado = "✅" if d["recall"] == 1.0 else "❌"
        print(f"{estado} {d['pregunta']}")
        print(f"   Esperado: {d['esperado']}")
        print(f"   Recuperados: {d['recuperados']}")
        print(f"   Recall@{TOP_K}: {d['recall']:.0%} | Precision@{TOP_K}: {d['precision']:.0%}\n")


if __name__ == "__main__":
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        golden_set = json.load(f)

    rag = RAGSystem()

    # Detalle del sistema híbrido (el que pide la consigna)
    print("=" * 80)
    print("DETALLE - RECUPERADOR HÍBRIDO")
    print("=" * 80)
    hibrido = evaluar(rag, golden_set, "hibrido")
    imprimir_detalle(hibrido)

    # Comparación de los tres modos
    print("=" * 80)
    print(f"{'Modo':<12}{'Recall@' + str(TOP_K):>12}{'Precision@' + str(TOP_K):>15}")
    print("-" * 39)
    for modo in ("bm25", "vectorial", "hibrido"):
        r = hibrido if modo == "hibrido" else evaluar(rag, golden_set, modo)
        print(f"{modo:<12}{r['recall']:>12.0%}{r['precision']:>15.0%}")
    print("=" * 80)