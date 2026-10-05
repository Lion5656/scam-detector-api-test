from backend.api.schemas.phone import PhoneReportRequest


def test_phone_report_request_accepts_connection_reason():
    req = PhoneReportRequest(
        phone_number="0912345678",
        phone_type="詐騙",
        connection_reason="同一批號碼前綴相同，且有相同回報者重複回報。",
    )

    assert req.connection_reason == "同一批號碼前綴相同，且有相同回報者重複回報。"
