from backend.services.phone_service.phone_service import PhoneService


def test_build_family_summary_filters_direct_related_and_keeps_target_phone_type():
    genealogy = {
        "scored_pairs": [
            {
                "left_phone": "0912000001",
                "right_phone": "0912000002",
                "score": 40,
                "reasons": ["直接電話轉介：A -> B"],
                "passed": True,
                "status": "accepted",
            },
            {
                "left_phone": "0912000001",
                "right_phone": "0912000003",
                "score": 25,
                "reasons": ["標籤一致"],
                "passed": False,
                "status": "rejected",
            },
            {
                "left_phone": "0912000001",
                "right_phone": "0912000005",
                "score": 30,
                "reasons": ["生命週期同步", "標籤一致"],
                "passed": True,
                "status": "accepted",
            },
            {
                "left_phone": "0912000002",
                "right_phone": "0912000004",
                "score": 60,
                "reasons": ["同號段且尾碼物理接近"],
                "passed": True,
                "status": "accepted",
            },
        ]
    }

    service = PhoneService()
    result = service._build_family_summary("0912000001", genealogy, "假投資")

    assert len(result) == 1
    assert result[0]["related_phone"] == "0912000002"
    assert result[0]["weight"] == 40
    assert result[0]["reason"] == "直接電話轉介：A -> B"
    assert result[0]["target_phone_type"] == "假投資"
    assert result[0]["status"] == "accepted"


def test_find_in_csv_records_matches_zero_padded_query():
    service = PhoneService()

    record = service._find_in_csv_records(
        "971000011",
        [{"phone_number": "0971000011", "status": "black"}],
    )

    assert record is not None
    assert record["phone_number"] == "0971000011"
