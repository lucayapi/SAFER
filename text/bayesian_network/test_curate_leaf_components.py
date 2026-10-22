from pathlib import Path

import pandas as pd

from curate_leaf_components import (
    build_outputs,
    load_definitions,
    recode,
)


CODEBOOK = Path(__file__).with_name("component_definitions_leaf.yaml")


def test_recode_uses_reported_content_and_role() -> None:
    units = pd.DataFrame([
        ("a1", "u1", "A0", "", "La victime travaille sur une toiture."),
        ("a1", "u2", "A1", "", "Il n'y avait pas de garde-corps en bordure."),
        ("a1", "u3", "B", "", "La victime a chuté du toit."),
        ("a1", "u4", "C", "", "La victime souffre d'une fracture et a été hospitalisée."),
        ("a2", "u5", "B", "", "Un camion benne tombe en panne."),
        ("a2", "u6", "C", "", "La lésion n'a pas nécessité son hospitalisation."),
    ], columns=["accident_id", "fact_id", "role", "topic_id", "sentence"])
    definitions = load_definitions(CODEBOOK)
    audited, matches = recode(units, definitions)
    found = matches.groupby("fact_id")["component_id"].agg(set).to_dict()
    assert "A0_ROOF" in found["u1"]
    assert "A1_FALL_GUARD" in found["u2"]
    assert "B_PERSON_FALL" in found["u3"]
    assert {"C_FRACTURE", "C_HOSPITALISATION"} <= found["u4"]
    assert "B_FALLING_OBJECT" not in found.get("u5", set())
    assert "C_HOSPITALISATION" not in found.get("u6", set())
    assert "B_PERSON_FALL" not in found["u1"]
    assert audited["fact_id"].is_unique


def test_accident_matrix_retains_empty_accident() -> None:
    definitions = load_definitions(CODEBOOK)
    units = pd.DataFrame([
        ("a1", "u1", "A0", "", "La victime travaille sur une toiture."),
        ("a2", "u2", "B", "", "Aucune expression reconnue."),
    ], columns=["accident_id", "fact_id", "role", "topic_id", "sentence"])
    audited, matches = recode(units, definitions)
    matrix, dictionary, audit = build_outputs(audited, matches, definitions)
    assert matrix["accident_id"].tolist() == ["a1", "a2"]
    assert matrix.loc[1, dictionary["factor_id"]].sum() == 0
    assert audit["accidents_without_component"] == 1
