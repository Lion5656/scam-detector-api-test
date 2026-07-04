import re

from pydantic import BaseModel, Field, model_validator


class PhoneQueryRequest(BaseModel):
    phone_number: str = Field(..., min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_phone_number_format(self):
        if not re.fullmatch(r"[0-9]{8,15}", self.phone_number):
            raise ValueError("電話號碼格式錯誤，請輸入 8~15 碼數字")
        return self


class PhoneReportRequest(BaseModel):
    phone_number: str = Field(..., min_length=1, max_length=20)
    phone_type: str = Field(..., min_length=1, max_length=50)
    other_type: str | None = Field(None, max_length=100)

    @model_validator(mode="after")
    def validate_phone_number_format(self):
        if not re.fullmatch(r"[0-9]{8,15}", self.phone_number):
            raise ValueError("電話號碼格式錯誤，請輸入 8~15 碼數字")
        return self

    @model_validator(mode="after")
    def validate_other_type(self):
        if self.phone_type == "其他" and not self.other_type:
            raise ValueError("phone_type 為 '其他' 時，必須提供 other_type")
        return self


class PhoneQueryResponse(BaseModel):
    phone_number: str
    status: str | None = None
    phone_type: str | None = None
    total_reports: int | None = None
    first_reported_at: str | None = None
    last_reported_at: str | None = None
    owner_name: str | None = None
    can_report: bool = False
    report_options: list[str] = []


class PhoneReportResponse(BaseModel):
    phone_number: str
    status: str
    total_reports: int
    report_time: str
    message: str
