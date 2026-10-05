import re
from typing import List, Tuple

from rapidfuzz import fuzz

# 弱訊號詞: 單獨出現時常見於正常付款、通知或客服問答，不應直接視為高風險
MONEY_RULE = re.compile(
    r"\$|\bUSD\b|TWD|NT|錢|金額|台幣|美金|人民幣|日圓|匯款|匯入|佣金|手續費|未繳清|未繳費|"
    r"交易連結|交易鏈結|入.{0,3}帳|出.{0,3}金|付.{0,3}費|付.{0,3}款|付.{0,3}現|"
    r"繳.{0,3}費|刷.{0,3}卡|退.{0,3}款|退.{0,3}費|轉.{0,3}帳"
)
OTP_RULE = re.compile(r"OTP|驗證信|驗證碼|動態密碼|授權碼|一次性密碼")
ACCOUNT_RULE = re.compile(r"帳號|銀行|帳戶")

# 高風險情境詞: 與詐騙語境強相關
URGENCY_RULE = re.compile(r"立即|馬上|立刻|趕快|限時|最後|逾期|否則|將停用|凍結|封鎖|24小時|今日截止|搶購|盡速前往")
INVEST_RULE = re.compile(r"(投資|飆股|內線|代操|保證獲利|高報酬|穩賺|跟單|當沖|虛擬貨幣|以太幣|比特幣|BTC|USDT)")
IMPERSONATE_RULE = re.compile(r"(客服|官方|有誤|升級|操作不當|操作異常|帳戶異常|系統異常|登入異常|風控|安全驗證|司法|檢警|刑事|警察|法院|傳票|偵查|金管會)")
SENSITIVE_RULE = re.compile(r"(身分證|登入密碼|網銀密碼|交易密碼|信用卡|卡號|三碼|CVV|戶頭|提款卡|存摺|個資|證件)")
GIFT_RULE = re.compile(r"(免費|免費LINE貼圖|中獎|抽獎|贈品|領取|免費領取|點我領|兌換|禮物卡|點數|Steam卡)")
JOB_RULE = re.compile(r"(打工|兼職|在家工作|日領|週領|無經驗|免經驗|高薪|刷流水|刷單|打字員)")
TIME_RULE = re.compile(r"\d+秒|\d+分|^\d{2}:\d{2}(\d{2})?$")

# 反詐騙及正規金融內容的白名單/降權規則
DEBUNK_RULE = re.compile(r"反詐騙|165|手法|破解|多問|多查|手法分析|防範|防詐|提醒民眾|MyGoPen|cofacts|宣導")
APR_RULE = re.compile(r"年百分率(\s*\d{1,2}(\.\d{1,2})?%\s*\~\s*\d{1,2}(\.\d{1,2})?%)?|總費用年百分率(\s*\d{1,2}(\.\d{1,2})?%\s*\~\s*\d{1,2}(\.\d{1,2})?%)?|APR")
CACL_RULE = re.compile(r"試算|交易稅|機動計息|本息攤還|每月還款金額|每日還款金額|收入負債比")
DISCLAIMER_RULE = re.compile(r"免責聲明|主管機關|保有核貸與否權利|審慎評估還款能力|實際貸款額度及利率|參閱|產品公開說明書")
PERMISSION_RULE = re.compile(r"免(自行)?上傳(財力|證明)|應用金融科技|授權本行")
NORMAL_PAYMENT_RULE = re.compile(
    r"付款成功|付款完成|完成付款|繳費完成|扣款成功|刷卡成功|交易成功|訂單|訂購|發票|"
    r"收據|消費通知|入帳通知|轉帳成功|退款完成|付款通知|帳單|繳款通知"
)
OTP_SAFETY_RULE = re.compile(r"請勿|不要|切勿|勿將|勿提供|不要提供|勿告知|請勿告知|本人操作|非本人操作|驗證碼僅供本人")
QUESTION_RULE = re.compile(r"請問|想問|想請教|是否|怎麼|如何|可以|能否|嗎|呢|\?+$|？+$")
BANK_BRAND_RULE = re.compile(r"銀行|本行|信用卡|客服專線|客服中心")
BANK_TXN_NOTIFICATION_RULE = re.compile(
    r"線上交易|交易驗證碼|本次交易|商店名稱|交易內容|完成付款|有效時間|"
    r"交易頁面|客服專線|如非本人操作|請確認交易內容"
)

WEAK_SIGNAL_RULES: List[Tuple[str, re.Pattern, int, str]] = [
    ("money", MONEY_RULE, 12, "金流交易相關字詞"),
    ("otp", OTP_RULE, 12, "驗證碼相關字詞"),
    ("account", ACCOUNT_RULE, 10, "帳戶金融相關字詞"),
]

STRONG_SIGNAL_RULES: List[Tuple[str, re.Pattern, int, str]] = [
    ("urgency", URGENCY_RULE, 35, "催促或時效壓力用詞"),
    ("invest", INVEST_RULE, 45, "投資獲利相關誘導字詞"),
    ("impersonate", IMPERSONATE_RULE, 55, "假冒官方客服或異常通知用詞"),
    ("sensitive", SENSITIVE_RULE, 45, "敏感金融個資相關用詞"),
    ("gift", GIFT_RULE, 35, "贈品中獎相關誘導用詞"),
    ("job", JOB_RULE, 30, "高薪兼職相關誘導用詞"),
    ("time", TIME_RULE, 20, "時間壓力或操作時限用詞"),
]

BANK_RULES: List[Tuple[re.Pattern, str]] = [
    (APR_RULE, "年百分率、總費用年百分率等相關用詞"),
    (CACL_RULE, "試算款項等相關用詞"),
    (DISCLAIMER_RULE, "免責聲明等相關用詞"),
    (PERMISSION_RULE, "授權與免責等相關用詞"),
]

# 僅保留辨識力較高的模糊詞，避免「驗證碼」這類正常場景常見詞直接加權
FUZZY_KEYWORDS = [
    "保證獲利",
    "高報酬",
    "帳戶異常",
    "虛擬貨幣",
    "代操",
    "會員指定任務",
    "抽獎機會",
    "數量有限",
    "最後機會",
    "認證失敗",
    "點我領取",
    "限時搶購",
    "立即行動",
    "官方客服",
    "系統升級",
    "司法調查",
    "金管會公告",
]


def compare_rules(text: str) -> Tuple[int, str, List[str]]:
    is_debunk = bool(len(re.findall(DEBUNK_RULE, text)) >= 2)
    if is_debunk:
        return 20, "，含反詐騙宣導用詞", []

    reason_list: List[str] = []
    match = 0
    for pattern, reason in BANK_RULES:
        counter = len(re.findall(pattern, text))
        if counter >= 2:
            match += 1
            reason_list.append(reason)
        if match == 2:
            return 20, "含正規銀行所常用之" + "、".join(reason_list), []

    bank_notification_markers = 0
    for pattern in (BANK_BRAND_RULE, BANK_TXN_NOTIFICATION_RULE, OTP_SAFETY_RULE, NORMAL_PAYMENT_RULE):
        if re.search(pattern, text):
            bank_notification_markers += 1

    if bank_notification_markers >= 3 and re.search(MONEY_RULE, text) and re.search(OTP_RULE, text):
        return 20, "含正規銀行交易驗證通知語境", []

    score = 0
    signal_hits: List[str] = []
    reason_list = []

    has_payment_context = bool(re.search(NORMAL_PAYMENT_RULE, text))
    has_otp_safety_context = bool(re.search(OTP_SAFETY_RULE, text))
    has_question_context = text.endswith(("?", "？")) or bool(re.search(QUESTION_RULE, text))

    strong_signal_count = 0

    for kw in FUZZY_KEYWORDS:
        similarity = fuzz.partial_ratio(kw, text)
        if similarity > 85:
            score += 18
            signal_hits.append(f"fuzzy:{kw}")
            reason_list.append(kw)

    for key, pattern, point, sentence in STRONG_SIGNAL_RULES:
        if re.findall(pattern, text):
            score += point
            strong_signal_count += 1
            signal_hits.append(key)
            reason_list.append(sentence)

    for key, pattern, point, sentence in WEAK_SIGNAL_RULES:
        if not re.findall(pattern, text):
            continue

        adjusted_point = point
        if key == "money" and (has_payment_context or has_question_context):
            adjusted_point = 5
        elif key == "otp" and (has_otp_safety_context or has_question_context):
            adjusted_point = 4
        elif key == "account" and has_question_context:
            adjusted_point = 4

        score += adjusted_point
        signal_hits.append(key)
        reason_list.append(sentence)

    low_signal_combo = {"money", "otp", "account"}.intersection(signal_hits)
    if strong_signal_count >= 1 and low_signal_combo:
        score += 18
        reason_list.append("金流或驗證資訊與高風險情境交互出現")

    if strong_signal_count == 0 and low_signal_combo:
        if has_payment_context or has_otp_safety_context:
            score = min(score, 18)
        elif has_question_context:
            score = min(score, 20)
        else:
            score = min(score, 28)

    dedup_hits = list(dict.fromkeys(signal_hits))
    dedup_reasons = list(dict.fromkeys(reason_list))
    reason = "疑似包含" + "、".join(dedup_reasons) if dedup_reasons else ""

    return min(score, 100), reason, dedup_hits
