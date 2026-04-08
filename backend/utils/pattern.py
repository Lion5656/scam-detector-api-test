import re
from typing import List, Tuple

from rapidfuzz import fuzz, process

# 建立關鍵字比對規則
MONEY_RULE = re.compile(r"\$|\bUSD\b|TWD|NT|錢|元|金額|台幣|美金|人民幣|日圓|匯款|匯入|佣金|手續費|未繳清|未繳費|交易連結|交易鏈結|入.{0,3}帳|出.{0,3}金|付.{0,3}費|付.{0,3}款|付.{0,3}現|繳.{0,3}費|刷.{0,3}卡|退.{0,3}款|退.{0,3}費|轉.{0,3}帳")
OTP_RULE = re.compile(r"OTP|驗證信|驗證碼|簡訊|動態密碼|授權碼|密碼|一次性密碼")
URGENCY_RULE = re.compile(r"立即|馬上|立刻|趕快|限時|最後|逾期|否則|將停用|凍結|封鎖|24小時|今日截止|搶購|盡速前往")
INVEST_RULE = re.compile(r"(投資|飆股|交易|內線|代操|保證獲利|高報酬|穩賺|跟單|當沖|虛擬貨幣|以太幣|比特幣|BTC|USDT)")
IMPERSONATE_RULE = re.compile(r"(客服|官方|有誤|升級|操作不當|操作異常|帳戶異常|系統異常|登入異常|風控|安全驗證|司法|檢警|刑事|警察|法院|傳票|偵查|金管會)")
SENSITIVE_RULE = re.compile(r"(身分證|密碼|帳號|銀行|信用卡|卡號|三碼|CVV|戶頭|提款卡|存摺|個資|證件)")
GIFT_RULE = re.compile(r"(免費|免費LINE貼圖|中獎|抽獎|贈品|領取|免費領取|點我領|兌換|禮物卡|點數|Steam卡)")
JOB_RULE = re.compile(r"(打工|兼職|在家工作|日領|週領|無經驗|免經驗|高薪|刷流水|刷單|打字員)")
TIME_RULE = re.compile(r"\d+秒|\d+分|^\d{2}:\d{2}(\d{2})?$")
DEBUNK_RULE = re.compile(r"反詐騙|165|手法|破解|多問|多查|手法分析|防範|防詐|提醒民眾|MyGoPen|cofacts|宣導")

RULES: List[Tuple[re.Pattern, int, str]] = [
    (MONEY_RULE, 30, "金錢及金錢交易相關誘導字詞"),
    (OTP_RULE, 30, "OTP驗證碼等相關誘導字詞"),
    (URGENCY_RULE, 50, "有催促及急促語氣"),
    (INVEST_RULE, 45, "投資及獲利異常相關誘導字詞"),
    (IMPERSONATE_RULE, 70, "假冒官方/客服及操作異常相關誘導用詞"),
    (SENSITIVE_RULE, 50, "索取個資及敏感資訊等相關用詞"),
    (GIFT_RULE, 40, "免費貼圖/贈品等相關誘導用詞"),
    (JOB_RULE, 30, "高時薪/輕鬆求職相關誘導用詞"),
    (TIME_RULE, 35, "時間指示操作相關指令用詞")
]

# 建立模糊比對高風險詞
FUZZY_KEYWORDS = ["保證獲利", "高報酬", "帳戶異常", "虛擬貨幣", "代操", "驗證碼", "中獎", "會員指定任務", "抽獎機會", "賣便貨", "數量有限", "最後機會", "認證失敗", "點我領取", "限時搶購", "立即行動", "官方客服", "系統升級", "帳戶安全", "司法調查", "金管會公告"]

# 比對關鍵字
def compare_rules(text: str) -> Tuple[int, str, List[str]]:
    # 規避反詐騙詞
    is_debunk = bool(re.search(DEBUNK_RULE, text))
    if is_debunk:
        return 20, "，含反詐騙宣導用詞", []
    
    score = 0
    reason_list = []
    hit_list = []
    # RapidFuzz 模糊比對，比對關鍵字相似度 (是否為子集)
    for kw in FUZZY_KEYWORDS:
        similarity = fuzz.partial_ratio(kw, text)
        if similarity > 85:
            if kw not in hit_list:
                score += 25
                hit_list.append(f"模糊命中: {kw}")
                reason_list.append(f"{kw}")
                
    # 正則比對
    for pattern, point, sentence in RULES:
        results = re.findall(pattern, text)
        if results:
            matches = min(len(results), 2) # 單一規則控制在最大分數 * 2 
            score += matches * point # 計算命中/次的分數
            reason_list.append(sentence)
            hit_list.extend(results)


    reason = "疑似包含" + ",".join(reason_list) if reason_list is not None else ""

    return min(score, 100), reason, hit_list          