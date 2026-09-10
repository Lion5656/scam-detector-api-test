import re

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, model_validator, ConfigDict
from pydantic.alias_generators import to_camel



class PhoneQueryRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )
    
    phone_number: str = Field(..., min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_phone_number_format(self):
        if not re.fullmatch(r"[0-9]{8,15}", self.phone_number):
            raise ValueError("電話號碼格式錯誤，請輸入 8~15 碼數字")
        return self


VALID_PHONE_TYPES = [
    "約會交友",
    "假投資",
    "假信貸",
    "假冒公務",
    "假包裹釣魚",
    "假求職",
    "商業騷擾",
    "假冒電商",
    "其他",
]


class PhoneReportRequest(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    phone_number: str = Field(..., min_length=1, max_length=20)
    phone_type: str = Field(..., min_length=1, max_length=50)
    other_type: str | None = Field(None, max_length=100)
    reporter_phone: str | None = Field(None, max_length=20)
    transfer_type: int | None = Field(default=0)
    transfer_content: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def validate_phone_number_format(self):
        if not re.fullmatch(r"[0-9]{8,15}", self.phone_number):
            raise ValueError("電話號碼格式錯誤，請輸入 8~15 碼數字")
        if self.reporter_phone and not re.fullmatch(r"[0-9]{8,15}", self.reporter_phone):
            raise ValueError("回報者電話格式錯誤，請輸入 8~15 碼數字")
        if self.transfer_type == 1 and self.transfer_content and not re.fullmatch(r"[0-9]{8,15}", self.transfer_content):
            raise ValueError("轉介類型為電話號碼時，轉介內容必須為 8~15 碼數字")
        return self

    @model_validator(mode="after")
    def validate_phone_type(self):
        if self.phone_type not in VALID_PHONE_TYPES:
            raise ValueError(
                "phone_type 必須為其中之一：約會交友、假投資、假信貸、假冒公務、假包裹釣魚、假求職、商業騷擾、假冒電商、其他"
            )
        if self.phone_type == "其他" and not self.other_type:
            raise ValueError("phone_type 為 '其他' 時，必須提供 other_type")
        if self.transfer_type not in (0, 1, 2, 3, 4, 9):
            raise ValueError("transfer_type 只能為 0、1、2、3、4、9，分別代表 無 / 電話 / LINE / URL / TG / 其他")
        if self.transfer_type == 0 and self.transfer_content:
            self.transfer_content = None
        return self


class PhoneQueryResponse(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    phone_number: str
    status: str | None = None
    phone_type: str | None = None
    total_reports: int | None = None
    first_reported_at: str | None = None
    last_reported_at: str | None = None
    owner_name: str | None = None
    can_report: bool = False
    # family analysis results (minimal but useful evidence)
    family_static: Any | None = Field(
        None,
        description=(
            "直接關聯證據列表；每筆包含 related_phone、weight、reason、target_phone_type。"
            "如果沒有關聯，則為空陣列。"
        ),
    )

class PhoneReportResponse(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )
    phone_number: str
    status: str
    total_reports: int
    report_time: str
    message: str
    connection_reason: str | None = None

class PhoneReportError(Exception):
    pass