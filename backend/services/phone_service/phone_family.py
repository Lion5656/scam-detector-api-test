"""以圖論分析通報資料，找出電話號碼之間的詐騙家族關聯。"""

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import networkx as nx

from backend.persistence.mysql_connection import engine as db_engine
from backend.repository.family_repository import family_repository
from backend.utils.phone_normalizer import _is_nan, normalize

logger = logging.getLogger(__name__)

__all__ = ["analyze_family", "build_phone_genealogy"]

MIN_LINK_SCORE = 40
SAME_REPORTER_WINDOW_SECONDS = 2 * 3600
TRANSFER_ASSET_LABELS = {2: "LINE", 3: "URL", 4: "TG"}
DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d",
)

# {(號碼A, 號碼B): [(通報者, 該時窗內兩號碼的通報次數), ...]}
ReportPairs = dict[tuple[str, str], list[tuple[str, int]]]


# --- 1. 基礎格式與工具 ---

def _safe_int(value: Any, default: int = 0) -> int:
    """轉成 int，NaN 或無法轉換時回傳預設值。"""
    if value is None or _is_nan(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _levenshtein(a: str, b: str) -> int:
    """計算字串編輯距離（用於尾數比對）。"""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _safe_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value or _is_nan(value) or str(value).lower() in ("none", "nan", ""):
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """把資料庫列（中文欄位）轉成評分用的標準格式；號碼無效時回傳 None。"""
    phone = normalize(record.get("電話號碼"))
    if not phone:
        return None

    transfer_type = _safe_int(record.get("轉介類型"), 0)
    raw_content = record.get("轉介內容")
    content = "" if raw_content is None or _is_nan(raw_content) else str(raw_content).strip()
    if content.lower() in ("nan", "none"):
        content = ""
    if transfer_type == 1 and content:
        content = normalize(content)

    return {
        "phone_number": phone,
        "reporter_id": normalize(record.get("通報者識別碼")),
        "total_reports": _safe_int(record.get("通報次數"), 1),
        "first_reported_at": _safe_datetime(record.get("首次通報時間")),
        "last_reported_at": _safe_datetime(record.get("最後通報時間")),
        "phone_type": str(record.get("通報標籤") or "其他"),
        "transfer_type": transfer_type,
        "transfer_content": content,
    }


# --- 2. 同通報者短時間內通報多支號碼 ---

def _list_report_events(phones: set[str]) -> dict[str, list[tuple[str, datetime]]]:
    """讀取指定號碼的通報事件，依通報者分組並按時間排序；查詢失敗時視為沒有事件。"""
    try:
        raw_events = family_repository.list_fraud_report_events(phones, db_engine)
    except Exception:
        logger.warning("讀取通報事件失敗，略過同通報者關聯", exc_info=True)
        return {}

    events: dict[str, list[tuple[str, datetime]]] = {}
    for reporter, entries in raw_events.items():
        parsed = [
            (phone, time)
            for phone, raw_time in entries
            if phone and (time := _safe_datetime(raw_time))
        ]
        if parsed:
            events[reporter] = sorted(parsed, key=lambda item: item[1])
    return events


def _build_report_pairs(events: dict[str, list[tuple[str, datetime]]]) -> ReportPairs:
    """找出同一通報者在 2 小時內通報的不同號碼配對，並記錄該時窗內的通報次數。"""
    report_pairs: ReportPairs = defaultdict(list)
    for reporter, reporter_events in events.items():
        for i, (left_phone, left_time) in enumerate(reporter_events):
            for right_phone, right_time in reporter_events[i + 1:]:
                if left_phone == right_phone:
                    continue
                if abs((right_time - left_time).total_seconds()) > SAME_REPORTER_WINDOW_SECONDS:
                    continue
                pair = (min(left_phone, right_phone), max(left_phone, right_phone))
                pair_report_count = sum(
                    1
                    for phone, time in reporter_events
                    if phone in pair
                    and abs((time - left_time).total_seconds()) <= SAME_REPORTER_WINDOW_SECONDS
                )
                report_pairs[pair].append((reporter, pair_report_count))
    return report_pairs


# --- 3. 核心權重計算引擎 ---

def _score_phone_pair(
    left: dict[str, Any], right: dict[str, Any], report_pairs: ReportPairs
) -> tuple[int, list[str], tuple[str, str] | None]:
    """依各項家族特徵計算兩支號碼的關聯分數。"""
    if left.get("phone_type") == "商業騷擾" or right.get("phone_type") == "商業騷擾":
        return 0, ["商業騷擾：禁止建立詐騙家族連結"], None

    score = 0
    reasons: list[str] = []
    directed_edge = None
    l_phone = left["phone_number"]
    r_phone = right["phone_number"]

    if left.get("transfer_type") == 1 and left.get("transfer_content") == r_phone:
        score += 40
        reasons.append("直接電話轉介：A -> B")
        directed_edge = (l_phone, r_phone)
    elif right.get("transfer_type") == 1 and right.get("transfer_content") == l_phone:
        score += 40
        reasons.append("直接電話轉介：B -> A")
        directed_edge = (r_phone, l_phone)

    if (
        left.get("transfer_type") == right.get("transfer_type")
        and left.get("transfer_type") not in (0, None)
        and left.get("transfer_content")
        and left.get("transfer_content") == right.get("transfer_content")
    ):
        score += 40
        reasons.append(f"外部導流端點共用 ({left.get('transfer_content')})")

    # 手機為 10 碼、家用市話（無區碼）為 8 碼，兩者都適用同號段比對，
    # 但長度不同時代表號碼類型不同，不應互相比對。
    same_segment = (
        l_phone[:6] == r_phone[:6] and len(l_phone) == len(r_phone) and len(l_phone) in (8, 10)
    )
    tail_l, tail_r = l_phone[-4:], r_phone[-4:]
    if (
        same_segment
        and tail_l.isdigit()
        and tail_r.isdigit()
        and (abs(int(tail_l) - int(tail_r)) <= 3 or _levenshtein(tail_l, tail_r) == 1)
    ):
        score += 30
        reasons.append("同號段且尾碼物理接近")

    pair_key = (min(l_phone, r_phone), max(l_phone, r_phone))
    pair_stats = report_pairs.get(pair_key, []) if l_phone and r_phone else []
    if pair_stats:
        supernode_strength = max((count for _, count in pair_stats), default=1)
        decay = 1.0 / (1.0 + math.log1p(supernode_strength))
        score += int(25 * decay)
        reasons.append("同通報者 2 小時內通報不同電話(已衰減)" if decay < 1 else "同通報者 2 小時內通報不同電話")

    l_first, r_first = left.get("first_reported_at"), right.get("first_reported_at")
    l_last, r_last = left.get("last_reported_at"), right.get("last_reported_at")
    if (
        l_first and r_first and l_last and r_last
        and abs((l_first - r_first).total_seconds()) <= 24 * 3600
    ):
        overlap = (min(l_last, r_last) - max(l_first, r_first)).total_seconds()
        union = (max(l_last, r_last) - min(l_first, r_first)).total_seconds()
        if union > 0 and overlap / union > 0.5:
            score += 15
            reasons.append("生命週期同步")

    ltag, rtag = left.get("phone_type"), right.get("phone_type")
    if ltag == rtag and ltag not in ("其他", "商業騷擾"):
        score += 15
        reasons.append(f"標籤一致({ltag})")
    elif {ltag, rtag} == {"約會交友", "假投資"}:
        score += 10
        reasons.append("跨標籤上下游產業鏈 (約會+投資)")

    return score, reasons, directed_edge


# --- 4. 分析流程與分桶機制 ---

def _list_candidate_pairs(processed: list[dict[str, Any]]) -> set[tuple[int, int]]:
    """以號碼、號段、導流端點、通報者與通報週分桶，只比對同桶內的紀錄。"""
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(processed):
        phone = record["phone_number"]
        buckets[f"num_{phone}"].append(idx)
        buckets[f"p_{phone[:6]}"].append(idx)
        if record["transfer_content"]:
            # 直接轉介以目標號碼分桶，讓 A -> B 仍能配對
            prefix = "num" if record["transfer_type"] == 1 else "t"
            buckets[f"{prefix}_{record['transfer_content']}"].append(idx)
        if record["reporter_id"]:
            buckets[f"r_{record['reporter_id']}"].append(idx)
        if record["first_reported_at"]:
            buckets[f"w_{record['first_reported_at'].strftime('%Y-W%U')}"].append(idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                candidate_pairs.add((min(left, right), max(left, right)))
    return candidate_pairs


def _summarize_family(
    family_id: str, graph: Any, nodes: set[str], processed: list[dict[str, Any]]
) -> dict[str, Any]:
    """彙整單一家族的號段、標籤、轉介鏈、共用導流資產與時間範圍。"""
    member_data = [r for r in processed if r["phone_number"] in nodes]
    paths = [
        f"{d['directed_edge'][0]} -> {d['directed_edge'][1]}"
        for _, _, d in graph.subgraph(nodes).edges(data=True)
        if d.get("directed_edge")
    ]
    assets: dict[str, set[str]] = defaultdict(set)
    for r in member_data:
        if r["transfer_type"] in TRANSFER_ASSET_LABELS:
            assets[TRANSFER_ASSET_LABELS[r["transfer_type"]]].add(r["transfer_content"])

    t_starts = [r["first_reported_at"] for r in member_data if r["first_reported_at"]]
    t_ends = [r["last_reported_at"] for r in member_data if r["last_reported_at"]]
    return {
        "family_id": family_id,
        "family_size": len(nodes),
        "core_prefixes": dict(Counter(n[:6] for n in nodes).most_common(3)),
        "primary_labels": dict(Counter(r["phone_type"] for r in member_data).most_common(3)),
        "transfer_chains": paths,
        "shared_assets": {k: list(v) for k, v in assets.items()},
        "timeline": {
            "start": min(t_starts).strftime("%Y-%m-%d %H:%M:%S") if t_starts else None,
            "end": max(t_ends).strftime("%Y-%m-%d %H:%M:%S") if t_ends else None,
        },
    }


def build_phone_genealogy(records: list[dict[str, Any]]) -> dict[str, Any]:
    """建立詐騙家族候選配對，並把相關號碼分組成家族。"""
    processed = [r for r in map(_normalize_record, records) if r is not None]
    skipped = len(records) - len(processed)

    # 只有 processed 內號碼的配對會被查詢，因此只需載入這些號碼的通報事件
    report_pairs = _build_report_pairs(_list_report_events({r["phone_number"] for r in processed}))
    candidate_pairs = _list_candidate_pairs(processed)

    edges = []
    scored_pairs = []
    graph = nx.DiGraph()
    for idx_a, idx_b in candidate_pairs:
        left = processed[idx_a]
        right = processed[idx_b]
        score, reasons, directed_edge = _score_phone_pair(left, right, report_pairs)
        passed = score >= MIN_LINK_SCORE

        pair_debug = {
            "left_phone": left["phone_number"],
            "right_phone": right["phone_number"],
            "left_type": left.get("phone_type"),
            "right_type": right.get("phone_type"),
            "left_reporter": left.get("reporter_id"),
            "right_reporter": right.get("reporter_id"),
            "left_transfer_type": left.get("transfer_type"),
            "right_transfer_type": right.get("transfer_type"),
            "left_transfer_content": left.get("transfer_content"),
            "right_transfer_content": right.get("transfer_content"),
            "score": score,
            "reasons": reasons,
            "passed": passed,
            "status": "accepted" if passed else "rejected",
        }
        scored_pairs.append(pair_debug)
        if not passed:
            continue

        edge_info = {
            "source": left["phone_number"],
            "target": right["phone_number"],
            "weight": score,
            "reasons": reasons,
            "directed_edge": directed_edge,
            "debug": pair_debug,
        }
        edges.append(edge_info)
        graph.add_edge(edge_info["source"], edge_info["target"], **edge_info)

    families = [
        _summarize_family(f"FAMILY_{i:03d}", graph, nodes, processed)
        for i, nodes in enumerate(nx.weakly_connected_components(graph), 1)
    ]

    return {
        "summary": {
            "input_records": len(records),
            "skipped_invalid_phone": skipped,
            "evaluated_records": len(processed),
            "evaluated_pairs": len(candidate_pairs),
            "total_edges": len(edges),
            "family_count": len(families),
        },
        "candidate_pairs": len(candidate_pairs),
        "edges": edges,
        "scored_pairs": scored_pairs,
        "families": families,
    }


# --- 5. 查詢號碼的直接家族關聯 ---

def analyze_family(phone_number: str, limit: int = 30) -> dict[str, Any]:
    """回傳與查詢號碼直接相連、且分數達門檻的家族關聯。"""
    normalized_phone = normalize(phone_number)
    result: dict[str, Any] = {
        "static": [],
        "cooccurrence": {"shared_reporters": [], "call_density": [], "content_similarity": []},
        "families": [],
        "phone_number": normalized_phone,
        "limit": limit,
    }

    records = family_repository.list_family_records(phone_number, engine=db_engine)
    if not records:
        return result

    genealogy = build_phone_genealogy(records)
    static = []
    seen = set()
    for pair in genealogy["scored_pairs"]:
        score = int(pair.get("score") or 0)
        if not pair.get("passed") or score < MIN_LINK_SCORE:
            continue

        left = str(pair.get("left_phone") or "")
        right = str(pair.get("right_phone") or "")
        if left == normalized_phone and right:
            related_phone = right
        elif right == normalized_phone and left:
            related_phone = left
        else:
            continue

        reason = "; ".join(pair.get("reasons", [])[:2]) or "同一家族成員"
        dedupe_key = (related_phone, score, reason)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        static.append({"related_phone": related_phone, "link_reason": reason, "weight": score})

    result["static"] = static[:limit]
    result["families"] = genealogy["families"]
    return result
