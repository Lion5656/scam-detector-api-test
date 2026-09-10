from __future__ import annotations

from typing import Any


def _is_nan(value: Any) -> bool:
    """安全判斷是否為 NaN（避免 float('nan') 這類 truthy 陷阱）"""
    try:
        return isinstance(value, float) and value != value
    except Exception:
        return False


def _pad_leading_zero(phone: str) -> str:
    """補回資料庫格式（去前導0）電話號碼的前導0。

    手機：10 碼完整格式，DB 存 9 碼（開頭為 9）。
    家用市話：9 碼完整格式，DB 存 8 碼。
    """
    if len(phone) == 9 and phone.startswith("9"):
        return f"0{phone}"
    if len(phone) == 8:
        return f"0{phone}"
    return phone


def _strip_taiwan_country_code(digits: str) -> str:
    """去除台灣國碼前綴（00886／886），還原成去前導 0 的國內碼格式。

    本地完整號碼一律以 0 開頭，去除國碼後的號碼不會有前導 0，
    因此兩者不會互相誤判。僅在去除國碼後長度為 8 或 9 碼
    （對應台灣手機／市話不含前導 0 的國內碼長度）時才轉換，
    避免誤傷其他無法辨識的號碼。轉換後交給 `_pad_leading_zero`
    統一補回前導 0，這裡不重複處理。
    """
    if digits.startswith("00886"):
        rest = digits[5:]
    elif digits.startswith("886"):
        rest = digits[3:]
    else:
        return digits
    return rest if len(rest) in (8, 9) else digits


def normalize(phone: Any) -> str:
    """Normalize a phone number to its complete display form (with leading zero)."""
    if phone is None or _is_nan(phone):
        return ""
    if isinstance(phone, float):
        if not phone.is_integer():
            # 不應該出現非整數電話號碼，視為髒資料
            return ""
        phone = int(phone)
    value = str(phone).strip()
    if value.lower() in ("nan", "none", ""):
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    digits = _strip_taiwan_country_code(digits)
    digits = _pad_leading_zero(digits)
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_db_phone(phone: Any) -> str:
    """轉成資料庫常見格式：手機 10 碼去前導 0 存 9 碼，家用市話 9 碼去前導 0 存 8 碼。"""
    normalized = normalize(phone)
    if normalized.startswith("0") and len(normalized) in (9, 10):
        return normalized[1:]
    return normalized
