import math
import re
from typing import Dict
from urllib.parse import parse_qs, urlparse

import pandas as pd
import tldextract
from pandas import DataFrame

FEATURE_NAMES = [
    "url_length", "num_dot", "num_dash", "num_slash",
    "is_https", "domain_length", "subdomain_depth", "has_digit_in_domain", 
    "num_dash_in_domain","num_phishing_keywords", "has_tw_keyword", 
    "path_length", "num_path_segments", "num_params", 
    "has_utm", "has_encoded_chars", "is_suspicious_tld",
    "domain_entropy", "is_real_gov", "is_fake_gov",
    "is_common_domain", "is_shortened", "is_common_tld"
]

SUSPICIOUS_TLD = ["xyz", "top", "cc", "tk", "ml", "ga", "club", "vip"]

TW_KEYWORDS = [
    "gov", "nhia", "健保", "郵局",
    "post", "etax", "tax", "電費",
    "taipower", "7-11", "familymart", "物流"
]

SHORTENER = ["bit.ly", "t.co", "tinyurl.com", "t.ly"]

COMMON_DOMAINS = {
    "google.com", "youtube.com", "facebook.com",
    "amazon.com", "wikipedia.org", "microsoft.com",
    "x.com"
}

COMMON_TLD = ["com", "org", "net", "tw", "com.tw"]

def calc_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)

def process_url_features(url: str) -> DataFrame:
    feat = extract_features(url)
    feat_list = [feat.get(name, 0) for name in FEATURE_NAMES]
    return pd.DataFrame([feat_list], columns=FEATURE_NAMES)

def extract_features(url: str) -> Dict:
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    scheme = parsed.scheme
    ext = tldextract.extract(url)
    root_domain = f"{ext.domain}.{ext.suffix}"  # google.com
    subdomain = ext.subdomain  
    tld = ext.suffix # com        

    features = dict()

    # 基礎特徵
    features['url_length'] = len(url)   
    features['num_dot'] = url.count('.')
    features['num_dash'] = url.count('-')
    features['num_slash'] = url.count('/')
    features['is_https'] = int(scheme == 'https')

    # domain特徵
    features['domain_length'] = len(domain)

    # subdomain深度
    features["subdomain_depth"] = len(subdomain.split('.')) if subdomain else 0
    
    # 檢設數字-字母替換
    features['has_digit_in_domain'] = int(bool(re.search(r'\d', domain)))
    features['num_dash_in_domain'] = domain.count('-')

    # 釣魚關鍵詞
    phishing_keywords = ['login', 'verify', 'secure', 'confirm', 'account', 
                        'authenticate', 'payment', 'receipt', 'identity', 
                        'update', 'exchange', 'security', 'auth', 'signin']
    phishing_count = sum(1 for kw in phishing_keywords if kw in url.lower())
    features['num_phishing_keywords'] = phishing_count

    url_lower = url.lower()
    features["has_tw_keyword"] = int(any(k in url_lower for k in TW_KEYWORDS))

    # path特徵
    features["path_length"] = len(path)
    features["num_path_segments"] = len([p for p in path.split("/") if p])

    # query特徵
    params = parse_qs(query)
    features["num_params"] = len(params)
    features["has_utm"] = int("utm_" in query)
    
    # URL編碼特徵（%字元表示編碼）
    features['has_encoded_chars'] = int('%' in url)

    # 可疑TLD(網域名稱的最後一部分)
    features["is_suspicious_tld"] = int(tld in SUSPICIOUS_TLD)
    
    # 計算entropy(混亂度)
    features["domain_entropy"] = calc_entropy(domain)
    
    # 真政府網址標註
    features["is_real_gov"] = 1 if domain.endswith("gov.tw") else 0

    # 正常網域
    features["is_common_domain"] = int(root_domain in COMMON_DOMAINS)

    # 短網域
    features["is_shortened"] = int(root_domain in SHORTENER)

    # 正常tld
    features["is_common_tld"] = int(tld in COMMON_TLD)

    return features
