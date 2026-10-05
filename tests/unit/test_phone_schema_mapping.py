from backend.services.phone_service.phone_service import PhoneService


def test_phone_lookup_uses_user_schema_columns():
    query = PhoneService._build_phone_lookup_query()

    assert "FROM phone p" in query
    assert "LEFT JOIN blacklist b ON p.phone_number = b.phone_number" in query
    assert "LEFT JOIN whitelist w ON p.phone_number = w.phone_number" in query
    assert "p.tags" in query
    assert "b.report_count" in query
    assert "w.org_name" in query


def test_status_and_phone_type_mapping():
    assert PhoneService._normalize_db_status("Black") == "black"
    assert PhoneService._normalize_db_status("White") == "white"
    assert PhoneService._normalize_db_status(None) == "unknown"
    assert PhoneService._normalize_db_phone_type("假投資") == "假投資"
