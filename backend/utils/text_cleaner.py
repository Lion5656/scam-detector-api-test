import re


# 標準化文字
def normalize_text(text: str) -> str:
    text = text.strip()
    # 清掉多餘的空白，只留一個
    text = re.sub(r"\s+", " ", text)
    return text