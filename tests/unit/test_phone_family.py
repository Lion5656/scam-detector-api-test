from backend.repository.family_repository import FamilyRepository
from backend.services.phone_service.phone_family import (
    analyze_family,
    build_phone_genealogy,
    _normalize,
)


def test_build_phone_genealogy_creates_weighted_family():
    records = [
        {
            "phone_number": "0912345001",
            "reporter_id": "0910000001",
            "total_reports": 12,
            "first_reported_at": "2024-01-01 08:00:00",
            "last_reported_at": "2024-01-02 08:00:00",
            "phone_type": "假投資",
            "transfer_type": 1,
            "transfer_content": "0912345007",
        },
        {
            "phone_number": "0912345003",
            "reporter_id": "0910000001",
            "total_reports": 8,
            "first_reported_at": "2024-01-01 09:00:00",
            "last_reported_at": "2024-01-02 09:00:00",
            "phone_type": "假投資",
            "transfer_type": 0,
            "transfer_content": "",
        },
        {
            "phone_number": "0912345007",
            "reporter_id": "0910000002",
            "total_reports": 3,
            "first_reported_at": "2024-01-01 10:00:00",
            "last_reported_at": "2024-01-03 10:00:00",
            "phone_type": "假投資",
            "transfer_type": 0,
            "transfer_content": "",
        },
    ]

    genealogy = build_phone_genealogy(records)

    assert genealogy["candidate_pairs"] >= 1
    assert genealogy["edges"]
    assert any(item["weight"] >= 60 for item in genealogy["edges"])
    assert genealogy["families"]
    assert genealogy["families"][0]["family_size"] >= 2


def test_phone_query_response_accepts_family_members_as_dicts():
    from backend.api.schemas.phone import PhoneQueryResponse

    payload = {
        "phone_number": "0912345001",
        "status": "black",
        "family_static": [
            {
                "phone_number": "0912345001",
                "related_phone": "0912345007",
                "relationship_type": "family_member",
                "link_reason": "同一詐騙家族成員",
            }
        ],
        "family_cooccurrence": {},
        "families": [],
    }

    response = PhoneQueryResponse(**payload)
    assert response.family_static[0]["related_phone"] == "0912345007"


def test_normalize_pads_leading_zero_for_9_digit_mobile():
    assert _normalize("974317261") == "0974317261"


def test_build_phone_genealogy_scores_same_prefix_after_padding_zero():
    records = [
        {
            "電話號碼": "974317261",
            "通報者識別碼": "A1",
            "通報次數": 1,
            "首次通報時間": "2024-01-01 08:00:00",
            "最後通報時間": "2024-01-02 08:00:00",
            "通報標籤": "假投資",
            "轉介類型": 0,
            "轉介內容": "",
        },
        {
            "電話號碼": "0974317262",
            "通報者識別碼": "A2",
            "通報次數": 1,
            "首次通報時間": "2024-01-03 08:00:00",
            "最後通報時間": "2024-01-04 08:00:00",
            "通報標籤": "其他",
            "轉介類型": 0,
            "轉介內容": "",
        },
    ]

    genealogy = build_phone_genealogy(records)

    pair = next(
        item
        for item in genealogy["scored_pairs"]
        if {item["left_phone"], item["right_phone"]} == {"0974317261", "0974317262"}
    )
    assert pair["score"] == 30
    assert "同號段且尾碼物理接近" in pair["reasons"]
    assert pair["passed"] is True


def test_analyze_family_returns_only_score_at_or_above_threshold():
    records = [
        {
            "電話號碼": "0912345001",
            "通報者識別碼": "A1",
            "通報次數": 12,
            "首次通報時間": "2024-01-01 08:00:00",
            "最後通報時間": "2024-01-02 08:00:00",
            "通報標籤": "假投資",
            "轉介類型": 1,
            "轉介內容": "0912345007",
        },
        {
            "電話號碼": "0912345003",
            "通報者識別碼": "A1",
            "通報次數": 8,
            "首次通報時間": "2024-01-01 09:00:00",
            "最後通報時間": "2024-01-02 09:00:00",
            "通報標籤": "假投資",
            "轉介類型": 0,
            "轉介內容": "",
        },
        {
            "電話號碼": "0912345007",
            "通報者識別碼": "A2",
            "通報次數": 3,
            "首次通報時間": "2024-01-01 10:00:00",
            "最後通報時間": "2024-01-03 10:00:00",
            "通報標籤": "假投資",
            "轉介類型": 0,
            "轉介內容": "",
        },
    ]

    genealogy = build_phone_genealogy(records)
    passed_scores = [pair["score"] for pair in genealogy["scored_pairs"] if pair["passed"]]
    assert passed_scores
    assert all(score >= 40 for score in passed_scores)
    result = analyze_family("0912345001")
    assert result["static"]
    assert all(item["weight"] >= 40 for item in result["static"])


def test_family_repository_loads_related_records():
    class FakeRowMapping:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

        def first(self):
            return self._rows[0] if self._rows else None

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return FakeRowMapping(self._rows)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            q = str(query)
            if "LIMIT 1" in q and "FROM blacklist" in q:
                return FakeResult([
                    {
                        "phone_number": "975000001",
                        "dominant_tag": "假求職",
                        "first_seen_at": "2024-01-01 07:00:00",
                        "last_seen_at": "2024-01-01 07:00:00",
                        "referral_type": 0,
                        "referral_info": "",
                    }
                ])
            if "phone_number NOT IN" in q:
                return FakeResult([
                    {
                        "phone_number": "975000002",
                        "phone_type": "假求職",
                        "first_reported_at": "2024-01-01 08:00:00",
                        "last_reported_at": "2024-01-02 08:00:00",
                        "transfer_type": 1,
                        "transfer_content": "0975000001",
                    }
                ])
            if "FROM fraud_reports" in q:
                return FakeResult([])
            return FakeResult([])

    class FakeEngine:
        def connect(self):
            return FakeConn()

    repository = FamilyRepository(FakeEngine())
    genealogy = repository.load_genealogy("0975000001")

    assert genealogy["families"]
    assert any(item["電話號碼"] == "975000002" for item in genealogy["records"])


def test_analyze_family_keeps_target_phone_in_db_genealogy(monkeypatch):
    from backend.services.phone_service import phone_family

    class FakeRowMapping:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

        def first(self):
            return self._rows[0] if self._rows else None

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return FakeRowMapping(self._rows)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            q = str(query)
            if "LIMIT 1" in q and "FROM blacklist" in q:
                return FakeResult([
                    {
                        "phone_number": "975000001",
                        "dominant_tag": "假求職",
                        "first_seen_at": "2024-01-01 07:00:00",
                        "last_seen_at": "2024-01-01 07:00:00",
                        "referral_type": 0,
                        "referral_info": "",
                    }
                ])
            if "phone_number NOT IN" in q:
                return FakeResult([
                    {
                        "phone_number": "975000002",
                        "phone_type": "假求職",
                        "first_reported_at": "2024-01-01 08:00:00",
                        "last_reported_at": "2024-01-02 08:00:00",
                        "transfer_type": 1,
                        "transfer_content": "0975000001",
                    }
                ])
            return FakeResult([])

    class FakeEngine:
        def connect(self):
            return FakeConn()

    monkeypatch.setattr(phone_family, "db_engine", FakeEngine())

    result = phone_family.analyze_family("0975000001")

    assert result["static"]
    assert any(item["related_phone"] == "0975000002" for item in result["static"])


def test_query_phone_normalizes_nine_digit_lookup(monkeypatch):
    from backend.services.phone_service.phone_service import PhoneService

    class FakeResult:
        def __init__(self, row):
            self._row = row

        def mappings(self):
            return self

        def first(self):
            return self._row

    class FakeConn:
        def __init__(self):
            self.executed = []

        def execute(self, query, params=None):
            self.executed.append(params)
            if params and params.get("phone_number") == "0912345001":
                return FakeResult({
                    "phone_number": "0912345001",
                    "tags": "Black",
                    "total_reports": 1,
                    "first_reported_at": "2024-01-01 08:00:00",
                    "last_reported_at": "2024-01-02 08:00:00",
                    "phone_type": "假投資",
                    "owner_name": "測試業者",
                })
            return FakeResult(None)

    class FakeBegin:
        def __enter__(self):
            return FakeConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setattr("backend.services.phone_service.phone_service.engine", FakeEngine())

    result = PhoneService().query_phone("912345001")

    assert result["phone_number"] == "0912345001"
    assert result["status"] == "black"


def test_build_phone_genealogy_caps_family_size():
    records = []
    for idx in range(12):
        phone = f"0912345{idx:03d}"
        records.append(
            {
                "phone_number": phone,
                "reporter_id": "0910000001",
                "total_reports": 6,
                "first_reported_at": "2024-01-01 08:00:00",
                "last_reported_at": "2024-01-02 08:00:00",
                "phone_type": "假投資",
                "transfer_type": 0,
                "transfer_content": "",
            }
        )

    for idx in range(11):
        records[idx]["transfer_type"] = 1 if idx < 11 else 0
        records[idx]["transfer_content"] = f"0912345{idx + 1:03d}" if idx < 11 else ""

    genealogy = build_phone_genealogy(records)
    assert genealogy["families"]
    assert all(family["family_size"] <= 10 for family in genealogy["families"])


def test_report_suspicious_updates_blacklist_and_reports(tmp_path, monkeypatch):
    import pandas as pd
    from backend.services.phone_service.phone_service import phone_service

    blacklist_path = tmp_path / "fraud_blacklist.csv"
    reports_path = tmp_path / "fraud_reports.csv"

    monkeypatch.setattr(phone_service, "CSV_FALLBACK_PATH", blacklist_path)
    monkeypatch.setattr(phone_service, "CSV_FALLBACK_CANDIDATES", (blacklist_path,))
    monkeypatch.setattr(phone_service, "REPORTED_CSV_PATH", reports_path)

    result = phone_service.report_suspicious(
        "0987654321",
        "假投資",
        reporter_phone="0912345678",
        connection_reason="可疑投資詐騙",
    )

    assert result["status"] == "black"
    assert blacklist_path.exists()
    blacklist_rows = pd.read_csv(blacklist_path)
    assert blacklist_rows.iloc[0]["電話號碼"] == "0987654321"
    assert blacklist_rows.iloc[0]["通報標籤"] == "假投資"
    assert blacklist_rows.iloc[0]["總回報次數"] == 1

    assert reports_path.exists()
    reports_rows = pd.read_csv(reports_path)
    assert reports_rows.iloc[0]["通報者門號"] == "0912345678"
    assert reports_rows.iloc[0]["電話號碼"] == "0987654321"


def test_load_fraud_report_events_uses_db_when_csv_missing(monkeypatch):
    from backend.services.phone_service import phone_family

    class FakeRowMapping:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return FakeRowMapping(self._rows)

    class FakeConn:
        def execute(self, query, params=None):
            return FakeResult([
                {"reporter_phone": "0910000001", "phone_number": "0912345001", "report_time": "2024-01-01 08:00:00"},
                {"reporter_phone": "0910000001", "phone_number": "0912345003", "report_time": "2024-01-01 09:30:00"},
            ])

    class FakeEngine:
        def connect(self):
            return FakeConn()

    monkeypatch.setattr(phone_family, "db_engine", FakeEngine())

    records = [
        {
            "電話號碼": "0912345001",
            "通報者識別碼": "0910000001",
            "通報次數": 1,
            "首次通報時間": "2024-01-01 08:00:00",
            "最後通報時間": "2024-01-01 08:00:00",
            "通報標籤": "假投資",
            "轉介類型": 0,
            "轉介內容": "",
        },
        {
            "電話號碼": "0912345003",
            "通報者識別碼": "0910000001",
            "通報次數": 1,
            "首次通報時間": "2024-01-01 09:30:00",
            "最後通報時間": "2024-01-01 09:30:00",
            "通報標籤": "假投資",
            "轉介類型": 0,
            "轉介內容": "",
        },
    ]

    genealogy = build_phone_genealogy(records)
    assert any(
        pair["passed"] and "同通報者 2 小時內通報不同電話" in "".join(pair["reasons"])
        for pair in genealogy["scored_pairs"]
    )


def test_analyze_family_handles_empty_db(monkeypatch):
    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            class M:
                def __init__(self, rows):
                    self._rows = rows

                def all(self):
                    return self._rows

                def first(self):
                    return self._rows[0] if self._rows else None

                def scalar(self):
                    return None

            return M(self._rows)


    class FakeConn:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, query, params=None):
            return FakeResult([])


    class FakeBegin:
        def __init__(self):
            pass

        def __enter__(self):
            return FakeConn([])

        def __exit__(self, exc_type, exc, tb):
            return False


    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setattr(
        "backend.services.phone_service.phone_family.engine",
        FakeEngine(),
    )

    result = analyze_family("0912345678")
    assert isinstance(result, dict)
    assert "static" in result and isinstance(result["static"], list)
    assert "cooccurrence" in result and isinstance(result["cooccurrence"], dict)
    assert result["static"] == []
    assert result["cooccurrence"] == {"shared_reporters": [], "call_density": [], "content_similarity": []}
