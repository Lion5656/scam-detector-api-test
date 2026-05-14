import re


# 標準化文字
def normalize_text(text: str) -> str:
    text = text.strip()
    # 清掉多餘的空白，只留一個
    text = re.sub(r"\s+", " ", text)
    # 清掉多餘的特殊字符
    text = re.sub(r"[－+\-*.]+", "", text)
    return text



def normalize_escape_sequences(text: str) -> str:
    text = text.replace('\\"', '"').replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    return text.strip()