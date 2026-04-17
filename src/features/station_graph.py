"""
src/features/station_graph.py
==============================
Grafo jerárquico de estaciones hidrológicas del Orinoco.

Define las relaciones de causalidad hidráulica entre estaciones:
el agua fluye de upstream a downstream, por lo que las estaciones
aguas arriba contienen información causal sobre el futuro de las
estaciones aguas abajo.

Ver: docs/STATION_TOPOLOGY.md para el grafo completo.
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# REGLA R8: Nomenclatura oficial de estaciones (NUNCA abreviar diferente)
STATION_ORDER: Dict[str, Dict] = {
    "ayacucho": {"km_from_source": 0, "order": 0, "role": "upstream_far"},
    "caicara": {"km_from_source": 500, "order": 1, "role": "upstream_mid"},
    "ciudad_bolivar": {"km_from_source": 900, "order": 2, "role": "upstream_near"},
    "palua": {"km_from_source": 950, "order": 3, "role": "target_default"},
}

# TODO: Implementar en Fase 1 (Feature Engineering)


def get_predictors(target: str, station_graph: Dict = STATION_ORDER) -> Dict:
    """Retorna las estaciones predictoras según posición relativa en el grafo.

    Reglas de causalidad:
        - Estaciones AGUAS ARRIBA → PREDICTORES PRIMARIOS (información causal)
        - El TARGET mismo → PREDICTOR AUTOREGRESIVO
        - Estaciones AGUAS ABAJO → EXCLUIDAS (el agua no fluye hacia arriba)

    Args:
        target: Nombre de la estación objetivo.
        station_graph: Diccionario con metadata de estaciones.

    Returns:
        Dict con keys 'primary' (list), 'self' (str), 'excluded' (list).
    """
    raise NotImplementedError("Implementar en Fase 1")
