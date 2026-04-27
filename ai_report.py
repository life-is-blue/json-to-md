#!/usr/bin/env python3
"""
ai_report.py — LLM-powered daily report + soul model builder + distillation pipeline.

Subcommands:
  report       Daily work report with precise stats
  push         Post latest report to WeCom group
  soul         Full-context observation extraction → SOUL.md
  lessons      Extract lessons learned → LESSONS.md
  distill      Distill SOUL + LESSONS → MEMORY.md rules
  gene-health  Compute Gene freshness, rebuild registry
  sync-memory  Commit and push ai-logs/ to remote

Config via .env (auto-loaded):
  LLM_API_KEY           API key (required for report/soul/lessons/distill)
  LLM_BASE_URL          OpenAI-compatible endpoint (default: https://api.openai.com/v1)
  LLM_MODEL_NAME        Model name (default: gpt-4o-mini)
  LLM_MAX_TOKENS        Max tokens for LLM response (default: 2000)
  WECOM_WEBHOOK_URL     WeCom group robot webhook (optional, for push)
  AI_LOGS_DIR           Log directory (default: ./ai-logs)
"""
import argparse, json, os, re, subprocess, sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from ai_engine import load_dotenv, call_engine, _codex_available
from ai_prompts import (
    REPORT_SYSTEM, SOUL_SYSTEM, DISTILL_SYSTEM, GROUNDING_SYSTEM,
    LESSONS_SYSTEM, SOUL_SKELETON, LESSONS_SKELETON, MEMORY_SKELETON,
)


load_dotenv()


def _ts_to_date(ts) -> date | None:
    """Parse meta.timestamp (int millis/seconds or ISO string) to a local-time date."""
    if ts is None or isinstance(ts, bool):
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000 if ts >= 1e12 else ts).date()
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()  # convert UTC/aware → local before taking date
        return dt.date()
    except (ValueError, OSError, OverflowError):
        return None


def session_days(path: Path) -> set[date]:
    """Every local date with at least one message. Mtime fallback if no timestamps found."""
    days: set[date] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = _ts_to_date((obj.get("meta") or {}).get("timestamp"))
                if d:
                    days.add(d)
    except OSError:
        pass
    if not days:
        try:
            days.add(datetime.fromtimestamp(path.stat().st_mtime).date())
        except OSError:
            pass
    return days


def find_sessions(logs_dir: Path, target_date: date = None) -> list[Path]:
    results = []
    for p in sorted(logs_dir.rglob("*.jsonl")):
        if "reports" in p.parts:
            continue
        if target_date and target_date not in session_days(p):
            continue
        results.append(p)
    return results


def extract_turns(path: Path, max_chars: int = 2000, target_date: date = None, tail: bool = False) -> str:
    turns, total = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            all_entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = obj.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                if target_date is not None:
                    d = _ts_to_date((obj.get("meta") or {}).get("timestamp"))
                    if d != target_date:
                        continue
                content = obj.get("content", "")
                if isinstance(content, list):
                    content = " ".join(i.get("text", "") for i in content if i.get("type") == "text")
                if not isinstance(content, str):  # defensive: handles null/int in malformed JSONL
                    continue
                # User turns get full 500 chars; assistant truncated to 200
                # User intent is the gold signal for both reports and soul modeling
                limit = 500 if role == "user" else 200
                entry = f"[{role}] {content[:limit]}"
                if tail:
                    all_entries.append(entry)
                else:
                    total += len(entry)
                    turns.append(entry)
                    if total > max_chars:
                        break
    except OSError:
        return ""
    if tail:
        # Take last N entries that fit within max_chars (bug fixes tend to be at tail)
        result, total = [], 0
        for entry in reversed(all_entries):
            total += len(entry)
            if total > max_chars:
                break
            result.append(entry)
        return "\n".join(reversed(result))
    return "\n".join(turns)



def quality_gate(observations: str) -> str:
    """Filter out low-signal observation bullets. Returns empty string if nothing survives."""
    REJECT_PATTERNS = [
        r"数据不足", r"无实质性", r"无法提取", r"样本有限",
        r"仅包含.*?/clear", r"仅包含.*?/resume", r"无实质性交互",
        r"需要更多.*?消息才能构建", r"(?:推测|初步判断|大概率)(?:使用|为|是)",
    ]
    lines = observations.strip().splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        if any(re.search(p, stripped) for p in REJECT_PATTERNS):
            continue
        # Bullets with <8 chars of actual content after stripping bold markers are noise
        text = re.sub(r"\*\*.*?\*\*[：:]?\s*", "", stripped.lstrip("- "))
        if len(text) < 8:
            continue
        kept.append(line)
    # If no bullet points survived, return empty
    if not any(l.strip().startswith("-") for l in kept):
        return ""
    return "\n".join(kept).strip()


def grounding_check(observations: str, user_turns: str) -> str:
    """LLM-as-judge: verify each observation bullet is grounded in actual user messages.
    Returns only GROUNDED bullets. Empty string if nothing survives.
    Pattern-key tags (<!-- pk: xxx -->) are stripped before grounding and re-attached after."""
    if not observations.strip() or not user_turns.strip():
        return ""

    # LLM fallback (small context) can't handle unbounded user_turns —
    # grounding requires observations + user_turns in one atomic call.
    # Codex exec (128K) handles full context; fallback truncates to 20K.
    if not _codex_available() and len(user_turns) > 20000:
        print(f"Grounding: user_turns truncated from {len(user_turns)} to 20000 (LLM fallback)", file=sys.stderr)
        user_turns = user_turns[:20000]

    # Strip pk tags before sending to grounding LLM — LLMs unreliably preserve HTML comments
    pk_re = re.compile(r'\s*<!--\s*pk:\s*[\w-]+\s*-->')
    pk_map = {}  # normalized bullet text → pk tag
    clean_lines = []
    for line in observations.strip().splitlines():
        pk_match = re.search(r'(<!--\s*pk:\s*[\w-]+\s*-->)', line)
        if pk_match and line.strip().startswith("-"):
            clean_text = pk_re.sub('', line).strip()
            norm_key = re.sub(r'\*\*.*?\*\*[：:]?\s*', '', clean_text.lstrip("- ")).strip()
            pk_map[norm_key] = pk_match.group(1)
            clean_lines.append(clean_text)
        else:
            clean_lines.append(line)
    clean_obs = "\n".join(clean_lines)

    prompt = f"## 观察\n\n{clean_obs}\n\n## 用户原始消息\n\n{user_turns}"
    verdict = call_engine(prompt, GROUNDING_SYSTEM)
    if not verdict:
        print("Grounding: LLM returned empty response (possible API refusal)", file=sys.stderr)
        return ""
    kept = []
    for line in verdict.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("GROUNDED:"):
            bullet = line[len("GROUNDED:"):].strip()
            if bullet:
                # Normalize: ensure single leading "- ", preserve content after it
                bullet = bullet.lstrip("-").lstrip()
                kept.append(f"- {bullet}")
        elif line.startswith("FABRICATED:"):
            print(f"Grounding rejected: {line[:100]}", file=sys.stderr)
        else:
            print(f"Grounding: unparseable judge line: {line[:80]}", file=sys.stderr)

    # Re-attach pk tags to surviving bullets
    for i, kept_line in enumerate(kept):
        bullet_text = re.sub(r'\*\*.*?\*\*[：:]?\s*', '', kept_line.lstrip("- ")).strip()
        for orig_norm, pk_tag in pk_map.items():
            if orig_norm in bullet_text or bullet_text in orig_norm:
                kept[i] = f"{kept_line} {pk_tag}"
                break

    if not kept:
        return ""
    return "\n".join(kept)


def observe_with_chunking(chunks: list[str]) -> str:
    """LLM observe — call_engine handles context limits internally."""
    combined = "\n\n---\n\n".join(chunks)
    return call_engine(combined, SOUL_SYSTEM)


def _get_required_fields(entry_text: str) -> list[str]:
    """Determine required fields based on type: tag in entry."""
    m = re.search(r'\|\s*type:\s*([\w-]+)', entry_text)
    entry_type = m.group(1) if m else "trap"
    if entry_type == "correction":
        return ["**误**", "**正**", "**因**"]
    if entry_type == "method":
        return ["**法**", "**步**", "**用**"]
    # trap / toolchain / arch / unknown → original fields
    return ["**坑**", "**因**", "**法**"]


def parse_lesson_entries(raw: str, target_date) -> list[dict]:
    """Parse LLM output into structured lesson entries.
    Each entry must have ## slug header + type-appropriate field triple."""
    entries = []
    parts = re.split(r'(?=^## [\w-]+$)', raw.strip(), flags=re.M)
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue
        m = re.match(r'^## ([\w-]+)\s*$', part.splitlines()[0])
        if not m:
            continue
        slug = m.group(1)
        # Type-aware triple validation
        missing = [f for f in _get_required_fields(part) if f not in part]
        if missing:
            print(f"Lessons: skipping {slug}, missing: {', '.join(missing)}", file=sys.stderr)
            continue
        # Fix date line: match "> anything | pk:" pattern precisely
        text = re.sub(
            r'^>\s*\d{4}-\d{2}-\d{2}\s*\|',
            f'> {target_date} |',
            part, count=1, flags=re.M
        )
        entries.append({"slug": slug, "text": text})
    return entries


def lessons_quality_gate(entries: list[dict]) -> list[dict]:
    """Mechanical filter for lesson entries — reject speculative or vague content."""
    REJECT_PATTERNS = [
        r"(?:推测|可能|大概|也许|似乎)(?:是|为|存在|导致)",
        r"(?:暂未验证|待确认|不确定)",
    ]
    kept = []
    for entry in entries:
        text = entry["text"]
        if any(re.search(p, text) for p in REJECT_PATTERNS):
            print(f"Lessons quality gate rejected: {entry['slug']}", file=sys.stderr)
            continue
        # correction without 因 is noise — root cause is the core value of a correction
        if 'type: correction' in text:
            if '**因**' not in text:
                print(f"Lessons quality gate rejected (correction without 因): {entry['slug']}", file=sys.stderr)
                continue
        kept.append(entry)
    return kept



def cmd_lessons(args):
    """Extract lessons learned from sessions into LESSONS.md."""
    logs_dir = Path(args.logs)
    lessons_path = Path(args.lessons)
    target_date = args.date or (date.today() - timedelta(days=1))

    sessions = find_sessions(logs_dir, target_date)
    if not sessions:
        print(f"No sessions for {target_date}", file=sys.stderr); return

    # Collect full day content — call_engine handles context limits
    chunks = []
    for s in sessions:
        excerpt = extract_turns(s, max_chars=200000, target_date=target_date)
        if excerpt:
            chunks.append(excerpt)
    if not chunks:
        print(f"No extractable content for {target_date}", file=sys.stderr); return

    combined = "\n\n---\n\n".join(chunks)
    system = LESSONS_SYSTEM.format(date=target_date)
    print(f"Lessons: {len(combined)//1024}KB input", file=sys.stderr)
    raw = call_engine(combined, system)

    if not raw or raw.strip() == "NONE":
        print(f"No lessons for {target_date}", file=sys.stderr); return

    entries = parse_lesson_entries(raw, target_date)
    entries = lessons_quality_gate(entries)
    if not entries:
        print(f"No valid lesson entries for {target_date}", file=sys.stderr); return

    # Dedup: skip entries whose slug already exists
    existing_slugs = set()
    if lessons_path.exists():
        for m in re.finditer(r'^## ([\w-]+)$', lessons_path.read_text(encoding="utf-8"), re.M):
            existing_slugs.add(m.group(1))

    new_entries = [e for e in entries if e["slug"] not in existing_slugs]
    if not new_entries:
        print(f"All lessons for {target_date} already exist", file=sys.stderr); return

    # Write
    if not lessons_path.exists():
        lessons_path.write_text(LESSONS_SKELETON.format(date=target_date, count=0), encoding="utf-8")
    content = lessons_path.read_text(encoding="utf-8")

    for entry in new_entries:
        # Insert absorbed:false marker after ## slug line
        text = re.sub(r'^(## [\w-]+)\n', r'\1\n<!-- absorbed: false -->\n', entry["text"], count=1, flags=re.M)
        content += f"\n{text}\n"

    # Update metadata
    entry_count = len(re.findall(r'^## [\w-]+$', content, re.M))
    content = re.sub(r'Entries: \d+', f'Entries: {entry_count}', content)
    content = re.sub(r'Last updated: \S+', f'Last updated: {target_date}', content)

    lessons_path.write_text(content, encoding="utf-8")
    print(f"OK {lessons_path} (+{len(new_entries)} lessons for {target_date})", file=sys.stderr)


def cmd_report(args):
    logs_dir = Path(args.logs)
    target_date = args.date or (date.today() - timedelta(days=1))
    sessions = find_sessions(logs_dir, target_date)
    reports_dir = logs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{target_date}.md"
    if not sessions:
        out_path.write_text(f"# {target_date}\n\n无 AI 会话记录。\n", encoding="utf-8")
        print(f"OK {out_path}", file=sys.stderr); return
    # Compute structured stats from session paths
    tool_counts, project_counts = {}, {}
    for s in sessions:
        try:
            rel = s.relative_to(logs_dir).parts
            tool, project = rel[0], rel[1] if len(rel) > 1 else "unknown"
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            project_counts[project] = project_counts.get(project, 0) + 1
        except (ValueError, IndexError):
            pass
    stats = f"## 精确统计（请直接引用，不要估算）\n\n"
    stats += f"**总 session 数: {len(sessions)}**\n\n"
    stats += "| 工具 | session 数 |\n|------|----------|\n"
    for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
        stats += f"| {t} | {c} |\n"
    stats += "\n| 项目 | session 数 |\n|------|----------|\n"
    for p, c in sorted(project_counts.items(), key=lambda x: -x[1]):
        stats += f"| {p} | {c} |\n"

    parts = []
    for s in sessions:
        try:
            tool = s.relative_to(logs_dir).parts[0]
        except (ValueError, IndexError):
            tool = "unknown"
        parts.append(f"## {tool}: {s.stem}\n{extract_turns(s, max_chars=4000, target_date=target_date)}")
    result = call_engine(f"Date: {target_date}\n\n{stats}\n\n## 会话详情\n\n" + "\n\n".join(parts), REPORT_SYSTEM)
    out_path.write_text(f"# {target_date}\n\n{result}\n", encoding="utf-8")
    print(f"OK {out_path}", file=sys.stderr)


def cmd_soul(args):
    logs_dir, soul_path = Path(args.logs), Path(args.soul)
    target_date = args.date  # None means "today's batch mode" (existing behavior)

    if target_date:
        # Date-specific mode: extract observations for one specific day
        sessions = find_sessions(logs_dir, target_date)
        if not sessions:
            print(f"No sessions for {target_date}", file=sys.stderr); return
        chunks = []
        for s in sessions:
            excerpt = extract_turns(s, max_chars=200000, target_date=target_date)
            if excerpt:
                chunks.append(excerpt)
        if not chunks:
            print(f"No extractable content for {target_date}", file=sys.stderr); return
        observations = observe_with_chunking(chunks)
        observations = quality_gate(observations)
        if not observations:
            print(f"Observations for {target_date} rejected by quality gate", file=sys.stderr); return
        # Layer 2: LLM grounding check — collect user turns with larger budget
        # Use 8000 mixed budget so ~4000 user-only chars survive filtering
        user_turns_text = ""
        for s in sessions:
            turns = extract_turns(s, max_chars=200000, target_date=target_date)
            user_turns_text += "\n".join(l for l in turns.splitlines() if l.startswith("[user]")) + "\n"
        observations = grounding_check(observations, user_turns_text)
        if not observations:
            print(f"Observations for {target_date} rejected by grounding check", file=sys.stderr); return
        entry_date = target_date
    else:
        # Existing batch mode (unchanged logic)
        today = date.today()
        if args.since:
            since_date = args.since
        elif soul_path.exists():
            since_date = datetime.fromtimestamp(soul_path.stat().st_mtime).date()
        else:
            since_date = date(2020, 1, 1)
        sessions = [s for s in find_sessions(logs_dir)
                    if max(session_days(s), default=date.min) >= since_date]
        if not sessions:
            print(f"No new sessions since {since_date}", file=sys.stderr); return
        chunks, total = [], 0
        for s in sessions:
            excerpt = extract_turns(s, max_chars=200000)
            if not excerpt:
                continue
            chunks.append(excerpt)
            total += len(excerpt)
            if total > 500000:  # soft cap for map-reduce across sessions
                break
        if not chunks:
            print("No extractable content from sessions", file=sys.stderr); return
        observations = observe_with_chunking(chunks)
        observations = quality_gate(observations)
        if not observations:
            print("Observations rejected by quality gate", file=sys.stderr); return
        # Layer 2: LLM grounding check — collect user turns with larger budget
        user_turns_text = ""
        for s in sessions:
            turns = extract_turns(s, max_chars=200000)
            user_turns_text += "\n".join(l for l in turns.splitlines() if l.startswith("[user]")) + "\n"
        observations = grounding_check(observations, user_turns_text)
        if not observations:
            print("Observations rejected by grounding check", file=sys.stderr); return
        entry_date = today

    # Count actual jsonl files on disk
    file_count = sum(1 for _ in logs_dir.rglob("*.jsonl") if "reports" not in _.parts)

    if not soul_path.exists():
        soul_path.write_text(SOUL_SKELETON.format(date=entry_date, count=file_count), encoding="utf-8")
    content = soul_path.read_text(encoding="utf-8")

    # Update metadata
    content = re.sub(r"Sessions:.*", f"Sessions: {file_count} files", content)
    content = re.sub(r"Last updated:.*", f"Last updated: {entry_date}", content)
    # Legacy format cleanup: remove "Sessions processed: N" if present
    content = re.sub(r"> Sessions processed: \d+\n", "", content)

    # Dedup: replace entry_date's entry if it exists, otherwise append
    date_header = f"\n### {entry_date}\n"
    entry = f"{date_header}<!-- absorbed: false -->\n\n{observations}\n"
    if date_header in content:
        segments = re.split(r'(?=\n### \d{4}-\d{2}-\d{2}\n)', content)
        content = "".join(s for s in segments if not s.startswith(date_header)) + entry
    else:
        content += entry

    soul_path.write_text(content, encoding="utf-8")
    print(f"OK {soul_path} ({entry_date}, +{len(sessions)} sessions)", file=sys.stderr)


def extract_unabsorbed(soul_path: Path) -> list[tuple[str, str]]:
    """Parse SOUL.md, return [(date_str, observation_text)] for unabsorbed entries."""
    if not soul_path.exists():
        return []
    content = soul_path.read_text(encoding="utf-8")
    entries = re.split(r'(?=\n### \d{4}-\d{2}-\d{2}\n)', content)
    result = []
    for entry in entries:
        m = re.match(r'\n### (\d{4}-\d{2}-\d{2})\n', entry)
        if not m:
            continue
        if "<!-- absorbed: true -->" in entry:
            continue
        date_str = m.group(1)
        # Strip the header and absorbed marker
        text = re.sub(r'^.*?-->\s*', '', entry[m.end():], count=1, flags=re.DOTALL).strip()
        if not text:
            text = entry[m.end():].strip()
        if text:
            result.append((date_str, text))
    return result


def extract_pattern_counts(soul_path: Path, lessons_path: Path | None = None) -> dict[str, int]:
    """Parse SOUL.md + LESSONS.md, count unique dates per pattern-key.

    Returns dict mapping pattern-key → number of distinct dates it appeared on.
    This mechanical count replaces unreliable LLM self-counting.
    """
    pk_dates: dict[str, set[str]] = {}
    pk_re = re.compile(r'<!--\s*pk:\s*([\w-]+)\s*-->')

    # --- SOUL.md: date sections with pk-tagged bullets ---
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8")
        entries = re.split(r'(?=\n### \d{4}-\d{2}-\d{2}\n)', content)
        for entry in entries:
            m = re.match(r'\n### (\d{4}-\d{2}-\d{2})\n', entry)
            if not m:
                continue
            date_str = m.group(1)
            for pk_match in pk_re.finditer(entry):
                key = pk_match.group(1)
                pk_dates.setdefault(key, set()).add(date_str)

    # --- LESSONS.md: each entry has `> YYYY-MM-DD | pk: xxx` ---
    if lessons_path and lessons_path.exists():
        content = lessons_path.read_text(encoding="utf-8")
        lesson_entries = re.split(r'(?=^## [\w-])', content, flags=re.M)
        date_pk_re = re.compile(r'>\s*(\d{4}-\d{2}-\d{2})\s*\|\s*pk:\s*([\w-]+)')
        for entry in lesson_entries:
            m = date_pk_re.search(entry)
            if m:
                pk_dates.setdefault(m.group(2), set()).add(m.group(1))

    return {k: len(v) for k, v in sorted(pk_dates.items(), key=lambda x: -len(x[1]))}


def mark_absorbed(soul_path: Path, dates: list[str]):
    """Mark observation entries as absorbed in SOUL.md."""
    if not soul_path.exists():
        return
    content = soul_path.read_text(encoding="utf-8")
    for d in dates:
        content = content.replace(
            f"### {d}\n<!-- absorbed: false -->",
            f"### {d}\n<!-- absorbed: true -->"
        )
    soul_path.write_text(content, encoding="utf-8")


def prune_old(soul_path: Path, keep_days: int = 30):
    """Remove absorbed entries older than keep_days from SOUL.md."""
    if not soul_path.exists():
        return
    content = soul_path.read_text(encoding="utf-8")
    cutoff = date.today() - timedelta(days=keep_days)
    segments = re.split(r'(?=\n### \d{4}-\d{2}-\d{2}\n)', content)
    kept = []
    pruned = 0
    for seg in segments:
        m = re.match(r'\n### (\d{4}-\d{2}-\d{2})\n', seg)
        if not m:
            kept.append(seg)
            continue
        entry_date = date.fromisoformat(m.group(1))
        if entry_date < cutoff and "<!-- absorbed: true -->" in seg:
            pruned += 1
            continue
        kept.append(seg)
    if pruned:
        soul_path.write_text("".join(kept), encoding="utf-8")
        print(f"Pruned {pruned} old absorbed entries from SOUL.md", file=sys.stderr)


def extract_unabsorbed_lessons(lessons_path: Path) -> list[tuple[str, str]]:
    """Parse LESSONS.md, return unabsorbed entries as (date_str, text) tuples.
    Entries without any absorbed marker are treated as unabsorbed (backward compat)."""
    if not lessons_path.exists():
        return []
    content = lessons_path.read_text(encoding="utf-8")
    entries = re.split(r'(?=^## [\w-]+$)', content, flags=re.M)
    result = []
    for entry in entries:
        entry = entry.strip()
        if not entry.startswith("## "):
            continue
        if "<!-- absorbed: true -->" in entry or "<!-- rejected:" in entry or "<!-- needs-review -->" in entry:
            continue
        m = re.search(r'>\s*(\d{4}-\d{2}-\d{2})\s*\|', entry)
        date_str = m.group(1) if m else "unknown"
        result.append((date_str, entry))
    return result


def mark_absorbed_lessons(lessons_path: Path, slugs: list[str]):
    """Mark lesson entries as absorbed in LESSONS.md.
    Handles both new format (replace absorbed:false→true) and legacy (insert marker)."""
    if not lessons_path.exists() or not slugs:
        return
    content = lessons_path.read_text(encoding="utf-8")
    for slug in slugs:
        escaped = re.escape(slug)
        # Try replacing absorbed:false → true (new format)
        new_content = re.sub(
            rf'^(## {escaped}\n)<!-- absorbed: false -->',
            rf'\1<!-- absorbed: true -->',
            content, count=1, flags=re.M
        )
        if new_content != content:
            content = new_content
            continue
        # Legacy: no marker at all → insert absorbed:true after ## slug line
        new_content = re.sub(
            rf'^(## {escaped}\n)(?!<!-- )',
            rf'\1<!-- absorbed: true -->\n',
            content, count=1, flags=re.M
        )
        content = new_content
    lessons_path.write_text(content, encoding="utf-8")


def prune_old_lessons(lessons_path: Path, keep_days: int = 90):
    """Remove absorbed lesson entries older than keep_days."""
    if not lessons_path.exists():
        return
    content = lessons_path.read_text(encoding="utf-8")
    cutoff = date.today() - timedelta(days=keep_days)
    entries = re.split(r'(?=^## [\w-]+$)', content, flags=re.M)
    kept, pruned = [], 0
    for entry in entries:
        if not entry.strip().startswith("## "):
            kept.append(entry)
            continue
        m = re.search(r'>\s*(\d{4}-\d{2}-\d{2})\s*\|', entry)
        if m and "<!-- absorbed: true -->" in entry:
            entry_date = date.fromisoformat(m.group(1))
            if entry_date < cutoff:
                pruned += 1
                continue
        kept.append(entry)
    if pruned:
        new_content = "".join(kept)
        # Update entry count
        entry_count = len(re.findall(r'^## [\w-]+$', new_content, re.M))
        new_content = re.sub(r'Entries: \d+', f'Entries: {entry_count}', new_content)
        lessons_path.write_text(new_content, encoding="utf-8")
        print(f"Pruned {pruned} old absorbed entries from LESSONS.md", file=sys.stderr)


def review_agent_entries(lessons_path: Path):
    """Review <!-- needs-review --> entries written by agent: apply quality gate,
    promote to absorbed:false or mark as rejected."""
    if not lessons_path.exists():
        return
    content = lessons_path.read_text(encoding="utf-8")
    if "<!-- needs-review -->" not in content:
        return
    entries = re.split(r'(?=^## [\w-]+$)', content, flags=re.M)
    reviewed = 0
    for i, entry in enumerate(entries):
        if "<!-- needs-review -->" not in entry:
            continue
        # Extract slug for logging
        slug_m = re.match(r'^## ([\w-]+)', entry.strip())
        slug = slug_m.group(1) if slug_m else "unknown"
        # Apply quality gate
        dummy = [{"slug": slug, "text": entry}]
        kept = lessons_quality_gate(dummy)
        if kept:
            entries[i] = entry.replace("<!-- needs-review -->", "<!-- absorbed: false -->")
            print(f"Lessons review: {slug} → approved", file=sys.stderr)
        else:
            today_str = date.today().isoformat()
            entries[i] = entry.replace("<!-- needs-review -->", f"<!-- rejected: {today_str} -->")
            print(f"Lessons review: {slug} → rejected", file=sys.stderr)
        reviewed += 1
    if reviewed:
        content = "".join(entries)
        lessons_path.write_text(content, encoding="utf-8")
        print(f"Reviewed {reviewed} agent-written entries in LESSONS.md", file=sys.stderr)


def parse_distill_ops(raw: str) -> list[tuple[str, str, str]]:
    """Parse LLM structured diff → [(op, section, content)]. Invalid lines skipped."""
    ops = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line == "NOP":
            continue
        m = re.match(r'^(ADD|STRENGTHEN|WEAKEN|REMOVE)\s+(MUST|MUST_NOT|PREFER|CONTEXT):\s*(.+)$', line)
        if m:
            ops.append((m.group(1), m.group(2), m.group(3)))
        elif line not in ("", "NOP"):
            print(f"Distill: unparseable line: {line[:80]}", file=sys.stderr)
    return ops


DISTILL_MIN_ENTRIES = 7  # minimum unabsorbed entries before auto-triggering distill


def _section_bounds(content: str, header: str) -> tuple[int, int] | None:
    """Return (start, end) line indices for a section's bullet area (exclusive of header)."""
    lines = content.splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.strip() == f"## {header}":
            start = i + 1
        elif start is not None and l.startswith("## "):
            return (start, i)
    if start is not None:
        return (start, len(lines))
    return None


def apply_ops(content: str, ops: list[tuple[str, str, str]]) -> str:
    """Apply structured diff operations to MEMORY.md content, scoped to target sections."""
    if not content.strip():
        content = MEMORY_SKELETON.format(date=date.today(), version=0)

    for op, section, payload in ops:
        header = "MUST NOT" if section == "MUST_NOT" else section
        section_re = rf'(## {re.escape(header)}\n)(.*?)(?=\n## |\Z)'

        if op == "ADD":
            def add_rule(m, _payload=payload):
                return m.group(1) + m.group(2).rstrip('\n') + f"\n- {_payload}\n"
            content = re.sub(section_re, add_rule, content, count=1, flags=re.DOTALL)

        elif op == "REMOVE":
            bounds = _section_bounds(content, header)
            if bounds:
                lines = content.splitlines()
                s, e = bounds
                kept = [l for l in lines[s:e] if not (l.lstrip().startswith("- ") and payload in l)]
                content = "\n".join(lines[:s] + kept + lines[e:]) + "\n"
            else:
                print(f"Distill: REMOVE section not found: {header}", file=sys.stderr)

        elif op == "STRENGTHEN":
            if "→" in payload:
                old_hint, new_rule = payload.split("→", 1)
                old_hint, new_rule = old_hint.strip(), new_rule.strip()
                bounds = _section_bounds(content, header)
                if bounds:
                    lines = content.splitlines()
                    s, e = bounds
                    replaced = False
                    for i in range(s, e):
                        if old_hint in lines[i]:
                            lines[i] = f"- {new_rule}"
                            replaced = True
                            break
                    if replaced:
                        content = "\n".join(lines) + "\n"
                    else:
                        print(f"Distill: STRENGTHEN target not found in {header}: {old_hint[:60]}", file=sys.stderr)

        elif op == "WEAKEN":
            bounds = _section_bounds(content, header)
            removed_line = None
            if bounds:
                lines = content.splitlines()
                s, e = bounds
                for i in range(s, e):
                    if lines[i].lstrip().startswith("- ") and payload in lines[i]:
                        removed_line = lines[i].lstrip("- ").strip()
                        lines.pop(i)
                        content = "\n".join(lines) + "\n"
                        break
            if removed_line:
                prefer_re = r'(## PREFER\n)(.*?)(?=\n## |\Z)'
                def add_weakened(m, _rl=removed_line):
                    return m.group(1) + m.group(2).rstrip('\n') + f"\n- {_rl} (待观察)\n"
                content = re.sub(prefer_re, add_weakened, content, count=1, flags=re.DOTALL)

    # Update metadata
    content = re.sub(r'Updated: \S+', f'Updated: {date.today()}', content)
    if (m := re.search(r'Version: (\d+)', content)):
        content = re.sub(r'Version: \d+', f'Version: {int(m.group(1)) + 1}', content)

    return content


def _auto_create_gene(genes_dir: Path, pk: str, area: str, pk_entries_text: list[str], today: date):
    """Auto-create Gene scaffold from LESSONS.md entries sharing a pk.

    If a type:method entry exists for this pk, extract approach from its **步** field.
    Otherwise create a scaffold with <!-- needs-review --> marker.
    """
    gene_dir = genes_dir / pk
    if gene_dir.exists():
        return  # idempotent

    # Check for method entries
    method_approach = None
    method_desc = None
    for text in pk_entries_text:
        if 'type: method' in text:
            fa_m = re.search(r'\*\*法\*\*[：:]\s*(.+)', text)
            bu_m = re.search(r'\*\*步\*\*[：:]\s*(.+?)(?=\n\*\*|\Z)', text, re.S)
            if fa_m:
                method_desc = fa_m.group(1).strip()
            if bu_m:
                method_approach = bu_m.group(1).strip()

    description = method_desc or f"<!-- TODO: needs human review — auto-scaffolded from {len(pk_entries_text)} entries -->"
    needs_review = method_desc is None

    # Create directories
    gene_dir.mkdir(parents=True, exist_ok=True)
    (gene_dir / "variants").mkdir(exist_ok=True)

    # Write gene.yaml (atomic)
    gene_yaml = (
        f"gene_id: GEN-{today.strftime('%Y%m%d')}-{pk[:3]}\n"
        f"name: {pk}\n"
        f"description: {description}\n"
        f"created: {today.isoformat()}\n"
        f"source_type: learning\n"
        f"context_tags: {area}\n"
        f"applicable_areas: {area}\n"
        f"usage_count: 0\n"
        f"decay_window_days: 90\n"
    )
    tmp = (gene_dir / "gene.yaml.tmp")
    tmp.write_text(gene_yaml, encoding="utf-8")
    os.replace(tmp, gene_dir / "gene.yaml")

    # Write variants/v1.yaml (atomic)
    approach = method_approach or "\n".join(f"  {i+1}. (from {slug})" for i, slug in enumerate(
        re.findall(r'^## ([\w-]+)', '\n'.join(pk_entries_text), re.M)[:5]
    ))
    v1_yaml = (
        f"version: 1\n"
        f"created: {today.isoformat()}\n"
        f"approach: |\n"
        f"  {approach}\n"
    )
    tmp_v = (gene_dir / "variants" / "v1.yaml.tmp")
    tmp_v.write_text(v1_yaml, encoding="utf-8")
    os.replace(tmp_v, gene_dir / "variants" / "v1.yaml")

    marker = " [needs-review]" if needs_review else ""
    print(f"Gene auto-created: {pk} (area={area}){marker}", file=sys.stderr)


def cmd_distill(args):
    """Distill SOUL.md observations + LESSONS.md lessons into MEMORY.md rules."""
    soul_path = Path(args.soul)
    memory_path = Path(args.memory)
    lessons_path = Path(args.lessons)

    # Phase 0: review agent-written entries (<!-- needs-review --> → quality gate)
    review_agent_entries(lessons_path)

    # Phase 1: extract unabsorbed from both sources
    unabsorbed_soul = extract_unabsorbed(soul_path)
    unabsorbed_lessons = extract_unabsorbed_lessons(lessons_path)
    all_unabsorbed = unabsorbed_soul + unabsorbed_lessons

    if not all_unabsorbed:
        print("Distill: no unabsorbed entries in SOUL.md or LESSONS.md", file=sys.stderr); return
    # Threshold applies to combined count (SOUL + LESSONS)
    if len(all_unabsorbed) < DISTILL_MIN_ENTRIES and not args.force:
        print(f"Distill: only {len(all_unabsorbed)} unabsorbed entries (need {DISTILL_MIN_ENTRIES}+). Use --force to override.", file=sys.stderr)
        return

    # Phase 1.5: mechanical pattern-key counting across ALL entries (absorbed + unabsorbed)
    pattern_counts = extract_pattern_counts(soul_path, lessons_path)
    pattern_section = ""
    if pattern_counts:
        # Include pk with count (days × occurrences) for richer signal
        strong_pks = {k: v for k, v in pattern_counts.items() if v >= 2}
        if strong_pks:
            lines = [f"  {k}: {v}天" for k, v in strong_pks.items()]
            pattern_section = (
                "\n\n## Pattern-Key 出现天数（机械统计，地面真值）\n\n"
                + "\n".join(lines)
            )
        print(f"Distill: {len(pattern_counts)} pattern-keys ({len(strong_pks)} with ≥2 days), top: "
              + ", ".join(f"{k}={v}" for k, v in list(pattern_counts.items())[:5]),
              file=sys.stderr)

    # Phase 2: read current MEMORY.md
    current_memory = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""

    # Phase 3: LLM → structured diff (call_engine handles context limits)
    obs_parts = []
    if unabsorbed_soul:
        obs_parts.append("### 行为观察（来自 SOUL.md）\n\n" +
                         "\n\n".join(f"#### {d}\n{t}" for d, t in unabsorbed_soul))
    if unabsorbed_lessons:
        obs_parts.append("### 经验教训（来自 LESSONS.md）\n\n" +
                         "\n\n".join(f"#### {d}\n{t}" for d, t in unabsorbed_lessons))
    obs_text = "\n\n".join(obs_parts)
    prompt = f"## Current MEMORY.md\n\n{current_memory}\n\n## New Input\n\n{obs_text}{pattern_section}"
    print(f"Distill: {len(all_unabsorbed)} entries ({len(prompt)//1024}KB prompt)", file=sys.stderr)
    raw_diff = call_engine(prompt, DISTILL_SYSTEM)

    # Phase 4: parse and apply
    ops = parse_distill_ops(raw_diff)
    if not ops:
        print("Distill: NOP, marking as absorbed", file=sys.stderr)
    else:
        new_memory = apply_ops(current_memory, ops)
        memory_path.write_text(new_memory, encoding="utf-8")

    # Phase 5: mark absorbed + prune old
    mark_absorbed(soul_path, [d for d, _ in unabsorbed_soul])
    prune_old(soul_path, keep_days=30)
    _mark_lessons_absorbed(lessons_path, unabsorbed_lessons)
    prune_old_lessons(lessons_path, keep_days=90)

    total = len(unabsorbed_soul) + len(unabsorbed_lessons)
    if ops:
        print(f"OK {memory_path} ({len(ops)} ops, {total} entries absorbed: "
              f"{len(unabsorbed_soul)} soul + {len(unabsorbed_lessons)} lessons)", file=sys.stderr)
    else:
        print(f"Distill: {total} entries marked absorbed (NOP)", file=sys.stderr)

    # Phase 6: Gene auto-extraction — pk≥3 days → create gene.yaml
    if pattern_counts:
        genes_dir = Path(args.logs) / ".genes"
        lesson_text_by_pk = {}  # pk → list of entry texts
        if lessons_path.exists():
            lc = lessons_path.read_text(encoding="utf-8")
            for entry in re.split(r'(?=^## [\w-])', lc, flags=re.M):
                pk_m = re.search(r'pk:\s*([\w-]+)', entry)
                if pk_m:
                    lesson_text_by_pk.setdefault(pk_m.group(1), []).append(entry)

        for pk, cnt in pattern_counts.items():
            if cnt < 3:
                continue
            area_m = None
            for text in lesson_text_by_pk.get(pk, []):
                area_m = re.search(r'area:\s*([\w-]+)', text)
                if area_m:
                    break
            area = area_m.group(1) if area_m else "unknown"
            entries = lesson_text_by_pk.get(pk, [])
            if entries:
                _auto_create_gene(genes_dir, pk, area, entries, date.today())

    # Phase 7: MEMORY.md health check (mechanical)
    if memory_path.exists():
        mem_counts = _count_memory_rules(memory_path)
        total_rules = sum(mem_counts.values())
        warnings = []

        if total_rules > 100:
            warnings.append(f"MEMORY.md 规模膨胀: {total_rules} 条规则 (>100)")

        prefer_count = mem_counts.get("PREFER", 0)
        must_count = mem_counts.get("MUST", 0)
        if prefer_count > max(15, int(must_count * 1.5)):
            warnings.append(f"PREFER ({prefer_count}) 显著多于 MUST ({must_count})，考虑升级强信号")

        unabsorbed_count = len(extract_unabsorbed_lessons(lessons_path))
        if unabsorbed_count > DISTILL_MIN_ENTRIES * 3:
            warnings.append(f"LESSONS 积压 {unabsorbed_count} 条未吸收")

        context_count = mem_counts.get("CONTEXT", 0)
        if context_count < 2 and total_rules > 20:
            warnings.append(f"CONTEXT section 仅 {context_count} 条，考虑补充环境约束")

        for w in warnings:
            print(f"Distill health: {w}", file=sys.stderr)


def _mark_lessons_absorbed(lessons_path: Path, unabsorbed_lessons: list[tuple[str, str]]):
    """Extract slugs from unabsorbed lessons and mark them absorbed."""
    slugs = []
    for _, text in unabsorbed_lessons:
        m = re.match(r'^## ([\w-]+)', text.strip())
        if m:
            slugs.append(m.group(1))
    mark_absorbed_lessons(lessons_path, slugs)


def _parse_gene_yaml(filepath: Path) -> dict | None:
    """Parse gene.yaml — flat top-level scalar fields only, no PyYAML.

    Skips comment lines, indented lines (block scalar content), and
    multiline block markers (value == '|'). Returns dict of scalar
    fields or None if file is empty/invalid.
    """
    if not filepath.is_file():
        return None
    result = {}
    for raw in filepath.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue
        # Indented line = block scalar content, skip
        if raw[0:1].isspace():
            continue
        # Top-level key: value
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            value = value.strip().strip('"').strip("'")
            if value == "|":
                continue  # block scalar marker — skip, content lines are indented
            result[key.strip()] = value
    return result or None


def cmd_gene_health(args):
    """Compute Gene freshness scores and output health report."""
    genes_dir = Path(args.genes_dir)
    if not genes_dir.is_dir():
        print(f"No genes directory at {genes_dir}", file=sys.stderr)
        return

    today = date.today()
    genes = []
    for entry in sorted(genes_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        gene_yaml = entry / "gene.yaml"
        gene = _parse_gene_yaml(gene_yaml)
        if not gene:
            continue
        gene["_name"] = entry.name
        gene["_path"] = str(gene_yaml)
        genes.append(gene)

    if not genes:
        print("No genes found", file=sys.stderr)
        return

    active, stale, degraded = [], [], []
    registry_entries = []
    for g in genes:
        last_used = g.get("last_used", "")
        decay_window = int(g.get("decay_window_days") or 90)
        if last_used:
            try:
                lu_date = date.fromisoformat(last_used[:10])
                days_since = (today - lu_date).days
            except ValueError:
                days_since = decay_window
        else:
            # Never used — check created date
            created = g.get("created", "")
            if created:
                try:
                    cr_date = date.fromisoformat(created[:10])
                    days_since = (today - cr_date).days
                except ValueError:
                    days_since = decay_window
            else:
                days_since = decay_window

        freshness = max(0.0, round(1.0 - days_since / decay_window, 3))
        status = "active" if freshness > 0.5 else "stale" if freshness > 0.2 else "degraded"
        g["_freshness"] = freshness
        g["_status"] = status

        # Update gene.yaml in place
        path = Path(g["_path"])
        content = path.read_text(encoding="utf-8")
        for field, val in [("freshness_score", freshness), ("decay_status", status)]:
            if re.search(rf"^{field}:", content, flags=re.M):
                content = re.sub(rf"^{field}:.*$", f"{field}: {val}", content, flags=re.M)
            else:
                content = content.rstrip("\n") + f"\n{field}: {val}\n"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)

        {"active": active, "stale": stale, "degraded": degraded}[status].append(g)
        registry_entries.append({
            "gene_id": g.get("gene_id", ""),
            "name": g["_name"],
            "path": g["_name"],
            "created": g.get("created", ""),
            "decay_status": status,
            "freshness_score": freshness,
        })

    # Rebuild registry.json from gene.yaml (SSOT: gene.yaml, registry is derived index)
    registry_path = genes_dir / "registry.json"
    tmp_registry = registry_path.with_suffix(".tmp")
    tmp_registry.write_text(json.dumps({"genes": registry_entries}, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_registry, registry_path)

    print(f"Gene Health: {len(active)} active, {len(stale)} stale, {len(degraded)} degraded", file=sys.stderr)
    for g in stale:
        print(f"  STALE: {g['_name']} (freshness={g['_freshness']}, last_used={g.get('last_used', 'never')})", file=sys.stderr)
    for g in degraded:
        print(f"  DEGRADED: {g['_name']} (freshness={g['_freshness']}, last_used={g.get('last_used', 'never')})", file=sys.stderr)


def cmd_push(args):
    """Push latest report to WeCom group webhook."""
    webhook = os.environ.get("WECOM_WEBHOOK_URL")
    if not webhook:
        print("WECOM_WEBHOOK_URL not set, skip push", file=sys.stderr); return
    reports_dir = Path(args.logs) / "reports"
    # Find the most recently modified work report (YYYY-MM-DD.md only, exclude daily-health-*)
    reports = sorted(
        [p for p in reports_dir.glob("*.md") if re.match(r'\d{4}-\d{2}-\d{2}\.md$', p.name)],
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not reports:
        print("No reports found", file=sys.stderr); return
    report_text = reports[0].read_text(encoding="utf-8")
    # WeCom markdown limit is 4096 bytes
    if len(report_text.encode("utf-8")) > 4000:
        report_text = report_text[:3500] + "\n\n...\n\n> 完整日报见服务器"
    body = json.dumps({"msgtype": "markdown", "markdown": {"content": report_text}}).encode()
    req = Request(webhook, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10):
            pass
        print(f"Pushed {reports[0].name} to WeCom", file=sys.stderr)
    except Exception as e:
        print(f"Push failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# cmd_daily — pure mechanical health report (no LLM)
# ---------------------------------------------------------------------------

def _tokenize_bigram(text: str) -> set[str]:
    """CJK bigram + Latin word tokenizer for Jaccard similarity."""
    # Extract Latin words and CJK characters
    tokens = re.findall(r'[一-鿿]|[a-zA-Z0-9_]+', text.lower())
    # Build bigrams from adjacent CJK characters
    result = set()
    cjk_buf = []
    for tok in tokens:
        if len(tok) == 1 and '一' <= tok <= '鿿':
            cjk_buf.append(tok)
        else:
            # Flush CJK buffer as bigrams
            for i in range(len(cjk_buf) - 1):
                result.add(cjk_buf[i] + cjk_buf[i + 1])
            cjk_buf = []
            if len(tok) > 1:
                result.add(tok)
    # Flush remaining CJK
    for i in range(len(cjk_buf) - 1):
        result.add(cjk_buf[i] + cjk_buf[i + 1])
    return result


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_all_lesson_pits(lessons_path: Path) -> list[tuple[str, str]]:
    """Extract (slug, pit_text) from ALL LESSONS.md entries (regardless of absorbed status)."""
    if not lessons_path.exists():
        return []
    content = lessons_path.read_text(encoding="utf-8")
    entries = re.split(r'(?=^## [\w-]+$)', content, flags=re.M)
    result = []
    for entry in entries:
        m = re.match(r'^## ([\w-]+)', entry.strip())
        if not m:
            continue
        slug = m.group(1)
        pit_m = re.search(r'\*\*坑\*\*[：:]\s*(.+)', entry)
        if pit_m:
            result.append((slug, pit_m.group(1).strip()))
    return result


def _count_memory_rules(memory_path: Path) -> dict[str, int]:
    """Count rules per section in MEMORY.md."""
    counts = {"MUST": 0, "MUST NOT": 0, "PREFER": 0, "CONTEXT": 0}
    if not memory_path.exists():
        return counts
    content = memory_path.read_text(encoding="utf-8")
    current_section = None
    for line in content.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            if section in counts:
                current_section = section
            else:
                current_section = None
        elif current_section and line.strip().startswith("- "):
            counts[current_section] += 1
    return counts


def _check_rule_freshness(memory_path: Path, soul_path: Path) -> list[tuple[str, str]]:
    """Check which MEMORY.md rules have recent evidence in SOUL.md (last 30 days).
    Uses pk tags as bridge. Returns [(rule_text, status)] where status is 'evidenced' or 'stale'."""
    if not memory_path.exists() or not soul_path.exists():
        return []
    # Collect pk tags from recent 30 days in SOUL.md
    cutoff = date.today() - timedelta(days=30)
    soul_content = soul_path.read_text(encoding="utf-8")
    recent_pks = set()
    entries = re.split(r'(?=\n### \d{4}-\d{2}-\d{2}\n)', soul_content)
    for entry in entries:
        m = re.match(r'\n### (\d{4}-\d{2}-\d{2})\n', entry)
        if not m:
            continue
        try:
            entry_date = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if entry_date >= cutoff:
            for pk_m in re.finditer(r'<!--\s*pk:\s*([\w-]+)\s*-->', entry):
                recent_pks.add(pk_m.group(1))

    # Check each rule against recent pks
    memory_content = memory_path.read_text(encoding="utf-8")
    results = []
    for line in memory_content.splitlines():
        line_s = line.strip()
        if not line_s.startswith("- "):
            continue
        rule_text = line_s[2:].strip()
        # Check if any recent pk keyword appears in rule text
        found = False
        for pk in recent_pks:
            # pk is kebab-case like "plan-before-act"; check each word
            for word in pk.split("-"):
                if len(word) > 2 and word.lower() in rule_text.lower():
                    found = True
                    break
            if found:
                break
        results.append((rule_text, "evidenced" if found else "stale"))
    return results


def cmd_daily(args):
    """Generate daily health report — pure mechanical analysis, no LLM."""
    logs_dir = Path(args.logs)
    target_date = args.date or date.today()
    soul_path = logs_dir / "SOUL.md"
    lessons_path = logs_dir / "LESSONS.md"
    memory_path = logs_dir / "MEMORY.md"
    genes_dir = logs_dir / ".genes"
    reports_dir = logs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"daily-health-{target_date}.md"

    sections = []
    todos = []

    # --- Section 1: 知识库摘要 ---
    s1 = [f"## 1. 知识库摘要\n"]
    # SOUL.md
    soul_total = soul_unabsorbed = soul_today = 0
    if soul_path.exists():
        sc = soul_path.read_text(encoding="utf-8")
        soul_total = len(re.findall(r'^### \d{4}-\d{2}-\d{2}$', sc, re.M))
        soul_unabsorbed = sc.count("absorbed: false")
        soul_today = len(re.findall(rf'^### {target_date}$', sc, re.M))
    # LESSONS.md
    les_total = les_absorbed = les_unabsorbed = les_review = 0
    if lessons_path.exists():
        lc = lessons_path.read_text(encoding="utf-8")
        les_total = len(re.findall(r'^## [\w-]+$', lc, re.M))
        les_absorbed = lc.count("absorbed: true")
        les_review = lc.count("needs-review")
        les_unabsorbed = lc.count("absorbed: false")
    # MEMORY.md
    mem_counts = _count_memory_rules(memory_path)
    mem_total = sum(mem_counts.values())
    # Genes
    gene_active = gene_stale = gene_degraded = 0
    reg_path = genes_dir / "registry.json"
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            for g in reg.get("genes", []):
                s = g.get("decay_status", "")
                if s == "active": gene_active += 1
                elif s == "stale": gene_stale += 1
                elif s == "degraded": gene_degraded += 1
        except (json.JSONDecodeError, KeyError):
            pass

    s1.append(f"| 知识库 | 总计 | 详情 |")
    s1.append(f"|--------|------|------|")
    s1.append(f"| SOUL.md | {soul_total} 条观察 | 今日 +{soul_today}, unabsorbed {soul_unabsorbed} |")
    s1.append(f"| LESSONS.md | {les_total} 条教训 | absorbed {les_absorbed}, unabsorbed {les_unabsorbed}, needs-review {les_review} |")
    s1.append(f"| MEMORY.md | {mem_total} 条规则 | MUST {mem_counts['MUST']}, MUST_NOT {mem_counts['MUST NOT']}, PREFER {mem_counts['PREFER']}, CONTEXT {mem_counts['CONTEXT']} |")
    s1.append(f"| .genes/ | {gene_active + gene_stale + gene_degraded} 个 Gene | active {gene_active}, stale {gene_stale}, degraded {gene_degraded} |")
    sections.append("\n".join(s1))
    if gene_stale:
        todos.append(f"审查 {gene_stale} 个 stale Gene")
    if gene_degraded:
        todos.append(f"处理 {gene_degraded} 个 degraded Gene")
    if les_review:
        todos.append(f"审查 {les_review} 条 needs-review 教训")

    # --- Section 2: 提升候选 ---
    s2 = ["## 2. 提升候选\n"]
    pattern_counts = extract_pattern_counts(soul_path, lessons_path)
    candidates = [(pk, cnt) for pk, cnt in pattern_counts.items() if cnt >= 3]
    if candidates:
        for pk, cnt in candidates:
            s2.append(f"- `{pk}` ({cnt} 天) → 可提取为 Gene: `scripts/extract-gene.sh {pk}`")
        todos.append(f"评估 {len(candidates)} 个 Gene 晋升候选")
    else:
        s2.append("无候选（需 pk ≥ 3 天）")
    sections.append("\n".join(s2))

    # --- Section 3: 潜在重复检测 ---
    s3 = ["## 3. 潜在重复检测\n"]
    pits = _parse_all_lesson_pits(lessons_path)
    duplicates = []
    for i in range(len(pits)):
        tokens_i = _tokenize_bigram(pits[i][1])
        for j in range(i + 1, len(pits)):
            sim = _jaccard(tokens_i, _tokenize_bigram(pits[j][1]))
            if sim >= 0.5:
                duplicates.append((pits[i][0], pits[j][0], round(sim, 2)))
    if duplicates:
        for a, b, sim in duplicates:
            s3.append(f"- `{a}` ↔ `{b}` (相似度 {sim:.0%})")
        todos.append(f"检查 {len(duplicates)} 对潜在重复教训")
    else:
        s3.append("未发现重复")
    sections.append("\n".join(s3))

    # --- Section 4: LESSONS 分布统计 ---
    s4 = ["## 4. LESSONS 分布统计\n"]
    month_counts: dict[str, int] = {}
    if lessons_path.exists():
        for m in re.finditer(r'>\s*(\d{4}-\d{2})-\d{2}\s*\|', lessons_path.read_text(encoding="utf-8")):
            ym = m.group(1)
            month_counts[ym] = month_counts.get(ym, 0) + 1
    if month_counts:
        s4.append("| 月份 | 新增 |")
        s4.append("|------|------|")
        for ym in sorted(month_counts):
            s4.append(f"| {ym} | {month_counts[ym]} |")
    else:
        s4.append("暂无数据")
    # High-value: pk≥3 days AND unabsorbed
    hv = [(pk, cnt) for pk, cnt in pattern_counts.items()
          if cnt >= 3 and any(pk in t for _, t in extract_unabsorbed_lessons(lessons_path))]
    if hv:
        s4.append(f"\n**高价值未吸收**: {', '.join(f'`{pk}`({cnt}天)' for pk, cnt in hv)}")
    sections.append("\n".join(s4))

    # --- Section 5: Gene 健康 ---
    s5 = ["## 5. Gene 健康\n"]
    if reg_path.exists():
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            genes = reg.get("genes", [])
            if genes:
                s5.append("| Gene | 状态 | 新鲜度 |")
                s5.append("|------|------|--------|")
                for g in genes:
                    s5.append(f"| {g.get('name', '?')} | {g.get('decay_status', '?')} | {g.get('freshness_score', '?')} |")
            else:
                s5.append("无 Gene")
        except (json.JSONDecodeError, KeyError):
            s5.append("registry.json 解析失败")
    else:
        s5.append("无 .genes/ 目录")
    sections.append("\n".join(s5))

    # --- Section 6: MEMORY.md 规则新鲜度 ---
    s6 = ["## 6. 规则新鲜度\n"]
    freshness = _check_rule_freshness(memory_path, soul_path)
    stale_rules = [(r, s) for r, s in freshness if s == "stale"]
    if freshness:
        evidenced_n = sum(1 for _, s in freshness if s == "evidenced")
        s6.append(f"- 有近期证据: {evidenced_n}/{len(freshness)}")
        s6.append(f"- 可能过时: {len(stale_rules)}/{len(freshness)}")
        if stale_rules:
            s6.append("\n**可能过时的规则**:")
            for r, _ in stale_rules[:5]:
                s6.append(f"- {r[:80]}...")
            if len(stale_rules) > 5:
                s6.append(f"- ...及其他 {len(stale_rules) - 5} 条")
            todos.append(f"审查 {len(stale_rules)} 条可能过时的规则")
    else:
        s6.append("MEMORY.md 为空或无规则")
    sections.append("\n".join(s6))

    # --- Section 7: 蒸馏链路健康 ---
    s7 = ["## 7. 蒸馏链路健康（最近 7 天）\n"]
    s7.append("| 日期 | Sessions | 日报 | SOUL | LESSONS |")
    s7.append("|------|----------|------|------|---------|")
    soul_content = soul_path.read_text(encoding="utf-8") if soul_path.exists() else ""
    lessons_content = lessons_path.read_text(encoding="utf-8") if lessons_path.exists() else ""
    for d in range(7):
        day = target_date - timedelta(days=d)
        n_sessions = len(find_sessions(logs_dir, day))
        has_report = (reports_dir / f"{day}.md").exists()
        has_soul = f"### {day}" in soul_content
        n_lessons = len(re.findall(rf'>\s*{day}\s*\|', lessons_content))
        s7.append(f"| {day} | {n_sessions} | {'✓' if has_report else '—'} | {'✓' if has_soul else '—'} | +{n_lessons} |")
    sections.append("\n".join(s7))

    # --- Section 8: 待办事项 ---
    s8 = ["## 8. 待办事项\n"]
    if todos:
        for i, t in enumerate(todos, 1):
            s8.append(f"{i}. {t}")
    else:
        s8.append("无待办 — 一切正常")
    sections.append("\n".join(s8))

    # Write report
    header = f"# Daily Health Report — {target_date}\n"
    content = header + "\n\n".join(sections) + "\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"OK {out_path}", file=sys.stderr)


def cmd_sync_memory(args):
    """Commit and push ai-logs/ (which IS the ai-memory repo) to remote.

    ai-logs/ is a git clone of the ai-memory repository. All cmd_* functions
    write directly into it. This command simply stages, commits, and pushes.
    No file copying — ai-logs/ is the SSOT.
    """
    logs_dir = Path(args.logs)
    git_dir = logs_dir / ".git"
    if not git_dir.is_dir():
        print(f"sync-memory: {logs_dir} is not a git repo (no .git/)", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run(["git", "add", "-A"], cwd=str(logs_dir), check=True,
                      capture_output=True, timeout=30)
        # Idempotent: skip if nothing changed
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(logs_dir),
                               capture_output=True, timeout=10)
        if result.returncode == 0:
            print("sync-memory: no changes to commit", file=sys.stderr); return
        today_str = date.today().isoformat()
        subprocess.run(["git", "commit", "-m", f"chore: sync {today_str}"],
                      cwd=str(logs_dir), check=True, capture_output=True, timeout=30)
        subprocess.run(["git", "push"], cwd=str(logs_dir), check=True,
                      capture_output=True, timeout=120)
        print(f"OK sync-memory: committed and pushed to ai-memory", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or "")
        print(f"sync-memory git error: {err[:200]}", file=sys.stderr); sys.exit(1)
    except subprocess.TimeoutExpired:
        print("sync-memory: git operation timed out", file=sys.stderr); sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="AI log report & soul builder")
    sub = p.add_subparsers(dest="cmd", required=True)
    default_logs = os.environ.get("AI_LOGS_DIR", "./ai-logs")
    r = sub.add_parser("report")
    r.add_argument("--date", type=date.fromisoformat, default=None)
    r.add_argument("--logs", default=default_logs)
    s = sub.add_parser("soul")
    s.add_argument("--date", type=date.fromisoformat, default=None)
    s.add_argument("--since", type=date.fromisoformat, default=None)
    s.add_argument("--logs", default=default_logs)
    s.add_argument("--soul", default=str(Path(default_logs) / "SOUL.md"))
    pu = sub.add_parser("push")
    pu.add_argument("--logs", default=default_logs)
    d = sub.add_parser("distill")
    d.add_argument("--logs", default=default_logs)
    d.add_argument("--soul", default=str(Path(default_logs) / "SOUL.md"))
    d.add_argument("--memory", default=str(Path(default_logs) / "MEMORY.md"))
    d.add_argument("--lessons", default=str(Path(default_logs) / "LESSONS.md"))
    d.add_argument("--force", action="store_true", help="Distill even with <7 entries")
    le = sub.add_parser("lessons")
    le.add_argument("--date", type=date.fromisoformat, default=None)
    le.add_argument("--logs", default=default_logs)
    le.add_argument("--lessons", default=str(Path(default_logs) / "LESSONS.md"))
    gh = sub.add_parser("gene-health")
    gh.add_argument("--genes-dir", default=str(Path(default_logs) / ".genes"))
    da = sub.add_parser("daily")
    da.add_argument("--logs", default=default_logs)
    da.add_argument("--date", type=date.fromisoformat, default=None)
    sm = sub.add_parser("sync-memory")
    sm.add_argument("--logs", default=default_logs)
    args = p.parse_args()
    {"report": cmd_report, "soul": cmd_soul, "push": cmd_push,
     "distill": cmd_distill, "lessons": cmd_lessons,
     "gene-health": cmd_gene_health, "daily": cmd_daily,
     "sync-memory": cmd_sync_memory}[args.cmd](args)


if __name__ == "__main__":
    main()
