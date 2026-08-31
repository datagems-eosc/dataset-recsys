from evaluation.mathe.evaluate_expert_reference import parse_material_ids


def test_parse_material_ids_handles_common_excel_values():
    assert parse_material_ids(None) == []
    assert parse_material_ids(199) == ["199"]
    assert parse_material_ids(199.0) == ["199"]
    assert parse_material_ids("44, 42, 43") == ["44", "42", "43"]
    assert parse_material_ids("281.282") == ["281", "282"]


def test_parse_material_ids_deduplicates_while_preserving_order():
    assert parse_material_ids("21, 22, 21") == ["21", "22"]
