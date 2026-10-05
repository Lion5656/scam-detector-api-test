import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

from backend.persistence.mysql_connection import engine as db_engine
from backend.repository.family_repository import family_repository
from backend.utils.phone_normalizer import _is_nan, normalize, normalize_db_phone

# --- 1. 基礎格式與工具 ---

_normalize = normalize
_normalize_db_phone = normalize_db_phone

__all__ = [
    "normalize",
    "normalize_db_phone",
    "_normalize",
    "_normalize_db_phone",
    "analyze_family",
    "build_phone_genealogy",
]


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert values to int while safely handling NaN or missing data."""
    if value is None or _is_nan(value):
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _levenshtein(a: str, b: str) -> int:
    """計算字串編輯距離 (用於尾數比對)"""
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            sub = prev[j-1] + (0 if ca == cb else 1)
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, sub)
        prev = cur
    return prev[lb]


def _safe_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime): return value
    if not value or _is_nan(value) or str(value).lower() in ("none", "nan", ""): return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try: return datetime.strptime(str(value), fmt)
        except ValueError: continue
    return None


def _load_fraud_report_events() -> Dict[str, List[Tuple[str, datetime]]]:
    """Load reporter-to-phone event history from the repository layer (DB only)."""
    events: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)

    db_events = family_repository.load_fraud_report_events(db_engine)
    for reporter, entries in db_events.items():
        normalized_entries: List[Tuple[str, datetime]] = []
        for phone, reported_at in entries:
            parsed_time = _safe_datetime(reported_at)
            if phone and parsed_time:
                normalized_entries.append((phone, parsed_time))
        if normalized_entries:
            events[reporter] = sorted(normalized_entries, key=lambda item: item[1])

    return {k: sorted(v, key=lambda item: item[1]) for k, v in events.items()}

# --- 2. 核心權重計算引擎 ---

def _score_phone_pair(left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[int, List[str], Optional[Tuple[str, str]]]:
    """Score pair strength across the supported scam-family heuristics."""
    if left.get("phone_type") == "商業騷擾" or right.get("phone_type") == "商業騷擾":
        return 0, ["商業騷擾：禁止建立詐騙家族連結"], None

    score = 0
    reasons = []
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

    if (left.get("transfer_type") == right.get("transfer_type") and 
        left.get("transfer_type") not in (0, None) and 
        left.get("transfer_content") == right.get("transfer_content") and 
        left.get("transfer_content")):
        score += 40
        reasons.append(f"外部導流端點共用 ({left.get('transfer_content')})")

    # 手機為 10 碼、家用市話（無區碼）為 8 碼，兩者都適用同號段比對，
    # 但長度不同時代表號碼類型不同，不應互相比對。
    if (
        l_phone[:6] == r_phone[:6]
        and len(l_phone) == len(r_phone)
        and len(l_phone) in (8, 10)
    ):
        tail_l, tail_r = l_phone[-4:], r_phone[-4:]
        try:
            dist = abs(int(tail_l) - int(tail_r))
            edit_dist = _levenshtein(tail_l, tail_r)
            if dist <= 3 or edit_dist == 1:
                score += 30
                reasons.append("同號段且尾碼物理接近")
        except: pass

    if left.get("phone_number") and right.get("phone_number"):
        pair_key = tuple(sorted((left["phone_number"], right["phone_number"])))
        pair_stats = getattr(_score_phone_pair, "_fraud_report_pairs", {}).get(pair_key, [])
        if pair_stats:
            supernode_strength = max((count for _, count in pair_stats), default=1)
            decay = 1.0 / (1.0 + math.log1p(supernode_strength))
            score += int(25 * decay)
            reasons.append("同通報者 2 小時內通報不同電話(已衰減)" if decay < 1 else "同通報者 2 小時內通報不同電話")
            if left.get("transfer_type") == 1 and left.get("transfer_content") == r_phone:
                directed_edge = (l_phone, r_phone)
            elif right.get("transfer_type") == 1 and right.get("transfer_content") == l_phone:
                directed_edge = (r_phone, l_phone)

    if left.get("first_reported_at") and right.get("first_reported_at") and left.get("last_reported_at") and right.get("last_reported_at"):
        start_diff = abs((left["first_reported_at"] - right["first_reported_at"]).total_seconds()) / 3600.0
        if start_diff <= 24:
            s1, e1 = left["first_reported_at"], left["last_reported_at"]
            s2, e2 = right["first_reported_at"], right["last_reported_at"]
            overlap = (min(e1, e2) - max(s1, s2)).total_seconds()
            union = (max(e1, e2) - min(s1, s2)).total_seconds()
            if union > 0 and (overlap / union) > 0.5:
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

# --- 3. 分析流程與分桶機制 ---

def build_phone_genealogy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build scam-family candidates and group related phone records."""
    processed = []
    skipped = 0
    for r in records:
        phone = _normalize(r.get("電話號碼"))
        if not phone:
            skipped += 1
            continue

        # Keep transfer-type coercion explicit so malformed values do not break scoring.
        ttype = _safe_int(r.get("轉介類型"), 0)

        raw_content = r.get("轉介內容")
        if raw_content is None or _is_nan(raw_content):
            content = ""
        else:
            content = str(raw_content).strip()
            if content.lower() in ("nan", "none"):
                content = ""

        if ttype == 1 and content:
            content = _normalize(content)

        processed.append({
            "phone_number": phone,
            "reporter_id": _normalize(r.get("通報者識別碼")),
            "total_reports": _safe_int(r.get("通報次數"), 1),
            "first_reported_at": _safe_datetime(r.get("首次通報時間")),
            "last_reported_at": _safe_datetime(r.get("最後通報時間")),
            "phone_type": str(r.get("通報標籤") or "其他"),
            "transfer_type": ttype,
            "transfer_content": content,
        })

    fraud_report_events = _load_fraud_report_events()
    fraud_report_pairs: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    for reporter, events in fraud_report_events.items():
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                left_phone, left_time = events[i]
                right_phone, right_time = events[j]
                if left_phone == right_phone:
                    continue
                delta_hours = abs((right_time - left_time).total_seconds()) / 3600.0
                if delta_hours <= 2:
                    pair = tuple(sorted((left_phone, right_phone)))
                    pair_report_count = sum(
                        1 for event_phone, event_time in events
                        if event_phone in pair and abs((event_time - left_time).total_seconds()) <= 7200
                    )
                    fraud_report_pairs[pair].append((reporter, pair_report_count))

    _score_phone_pair._fraud_report_pairs = fraud_report_pairs

    buckets = defaultdict(list)
    for idx, r in enumerate(processed):
        if r["phone_number"]:
            buckets[f"num_{r['phone_number']}"].append(idx)
            buckets[f"p_{r['phone_number'][:6]}"].append(idx)
        if r["transfer_content"]:
            # Direct referrals are bucketed by the target phone so A -> B can still match.
            if r["transfer_type"] == 1:
                buckets[f"num_{r['transfer_content']}"].append(idx)
            else:
                buckets[f"t_{r['transfer_content']}"].append(idx)
        if r["reporter_id"]: buckets[f"r_{r['reporter_id']}"].append(idx)
        if r["first_reported_at"]:
            week_key = r["first_reported_at"].strftime("%Y-W%U")
            buckets[f"w_{week_key}"].append(idx)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2: continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidate_pairs.add(tuple(sorted((members[i], members[j]))))

    edges = []
    scored_pairs = []
    graph = nx.DiGraph() if nx else None
    
    for idx_a, idx_b in candidate_pairs:
        left = processed[idx_a]
        right = processed[idx_b]
        score, reasons, d_edge = _score_phone_pair(left, right)
        passed = score >= 40

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

        if passed:
            edge_info = {
                "source": left["phone_number"],
                "target": right["phone_number"],
                "weight": score,
                "reasons": reasons,
                "directed_edge": d_edge,
                "debug": pair_debug,
            }
            edges.append(edge_info)
            if graph is not None:
                graph.add_edge(edge_info["source"], edge_info["target"], **edge_info)

    families = []
    if nx and graph:
        components = list(nx.weakly_connected_components(graph))
        for i, nodes in enumerate(components, 1):
            sub = graph.subgraph(nodes)
            member_data = [r for r in processed if r["phone_number"] in nodes]
            
            prefixes = Counter([n[:6] for n in nodes])
            tags = Counter([r["phone_type"] for r in member_data])
            paths = [f"{d['directed_edge'][0]} -> {d['directed_edge'][1]}" 
                     for _, _, d in sub.edges(data=True) if d.get('directed_edge')]
            
            assets = defaultdict(set)
            for r in member_data:
                if r["transfer_type"] in (2, 3, 4):
                    assets[{2: "LINE", 3: "URL", 4: "TG"}.get(r["transfer_type"], "其他")].add(r["transfer_content"])

            t_starts = [r["first_reported_at"] for r in member_data if r["first_reported_at"]]
            t_ends = [r["last_reported_at"] for r in member_data if r["last_reported_at"]]

            families.append({
                "family_id": f"FAMILY_{i:03d}",
                "family_size": len(nodes),
                "core_prefixes": dict(prefixes.most_common(3)),
                "primary_labels": dict(tags.most_common(3)),
                "transfer_chains": paths,
                "shared_assets": {k: list(v) for k, v in assets.items()},
                "timeline": {
                    "start": min(t_starts).strftime("%Y-%m-%d %H:%M:%S") if t_starts else None,
                    "end": max(t_ends).strftime("%Y-%m-%d %H:%M:%S") if t_ends else None
                }
            })

    return {
        "summary": {
            "input_records": len(records),
            "skipped_invalid_phone": skipped,
            "evaluated_records": len(processed),
            "evaluated_pairs": len(candidate_pairs),
            "total_edges": len(edges),
            "family_count": len(families)
        },
        "candidate_pairs": len(candidate_pairs),
        "edges": edges,
        "scored_pairs": scored_pairs,
        "families": families
    }

# --- 4. DB-backed family lookup ---

# This path prefers the real MySQL tables and keeps the second-stage filtering aligned with the
# normalized phone format used across the application.
def _load_phone_genealogy_from_db(phone_number: str) -> Dict[str, Any]:
    """Load related scam-family data from the repository layer."""
    engine_candidate = db_engine if db_engine is not None else None
    return family_repository.load_genealogy(phone_number, engine=engine_candidate)


def analyze_family(phone_number: str, limit: int = 30):
    """Return accepted direct-family links for a phone number using the DB-backed genealogy."""
    normalized_phone = _normalize(phone_number)
    genealogy = _load_phone_genealogy_from_db(phone_number)
    if genealogy.get("error"):
        return {
            "static": [],
            "cooccurrence": {"shared_reporters": [], "call_density": [], "content_similarity": []},
            "families": [],
            "phone_number": normalized_phone,
            "limit": limit,
        }

    static = []
    seen = set()
    for pair in genealogy.get("scored_pairs", []) or []:
        score = int(pair.get("score") or 0)
        if not pair.get("passed") or score < 40:
            continue
        left = str(pair.get("left_phone") or "")
        right = str(pair.get("right_phone") or "")
        related_phone = None
        reason = "; ".join(pair.get("reasons", [])[:2]) or "同一家族成員"

        if left == normalized_phone and right:
            related_phone = right
        elif right == normalized_phone and left:
            related_phone = left

        if not related_phone:
            continue

        dedupe_key = (related_phone, score, reason)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        static.append({
            "related_phone": related_phone,
            "link_reason": reason,
            "weight": score,
        })

    result = {
        "static": static[:limit],
        "cooccurrence": {"shared_reporters": [], "call_density": [], "content_similarity": []},
        "families": genealogy.get("families", []),
        "phone_number": normalized_phone,
        "limit": limit,
    }
    return result