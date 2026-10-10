from src.pipeline.entity_resolver import EntityResolver


def test_entity_resolution_maps_company_alias_to_security_master():
    entities = EntityResolver().resolve(
        "The Boeing Company senior debt was placed on credit watch."
    )

    assert entities
    assert entities[0].ticker == "BA"
    assert entities[0].company_name == "The Boeing Company"


def test_entity_resolution_leaves_broad_macro_event_unlinked():
    assert EntityResolver().resolve(
        "The Federal Reserve raised rates as inflation accelerated."
    ) == []
