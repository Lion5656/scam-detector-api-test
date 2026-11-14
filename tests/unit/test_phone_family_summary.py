import importlib

from backend.services.phone_service.phone_service import PhoneService


def test_list_family_links_filters_weak_links_and_keeps_target_phone_type(monkeypatch):
    phone_service_module = importlib.import_module("backend.services.phone_service.phone_service")

    def fake_analyze_family(phone_number):
        assert phone_number == "0912000001"
        return {
            "static": [
                {"related_phone": "912000002", "link_reason": "直接電話轉介：A -> B", "weight": 40},
                {"related_phone": "0912000002", "link_reason": "直接電話轉介：A -> B", "weight": 40},
                {"related_phone": "0912000005", "link_reason": "生命週期同步", "weight": 30},
            ]
        }

    monkeypatch.setattr(phone_service_module, "analyze_family", fake_analyze_family)

    result = PhoneService()._list_family_links("0912000001", "假投資")

    assert result == [
        {
            "related_phone": "0912000002",
            "weight": 40,
            "reason": "直接電話轉介：A -> B",
            "target_phone_type": "假投資",
        }
    ]


def test_list_family_links_returns_empty_when_analysis_fails(monkeypatch):
    phone_service_module = importlib.import_module("backend.services.phone_service.phone_service")

    calls = []

    def failing_analyze_family(phone_number):
        calls.append(phone_number)
        raise RuntimeError("db down")

    monkeypatch.setattr(phone_service_module, "analyze_family", failing_analyze_family)

    assert PhoneService()._list_family_links("0912000001", "假投資") == []
    assert calls == ["0912000001"]


def test_find_in_csv_records_matches_zero_padded_query():
    service = PhoneService()

    record = service._find_in_csv_records(
        "971000011",
        [{"phone_number": "0971000011", "status": "black"}],
    )

    assert record is not None
    assert record["phone_number"] == "0971000011"
