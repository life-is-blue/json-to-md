"""Tests for ai_report.py core functions.

Tests call production code directly — no logic duplication.
"""

import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

# Ensure we can import from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_report import (
    parse_distill_ops,
    apply_ops,
    quality_gate,
    parse_lesson_entries,
    _parse_gene_yaml,
    _tokenize_bigram,
    _jaccard,
    _parse_all_lesson_pits,
    _count_memory_rules,
    extract_pattern_counts,
)
from ai_prompts import MEMORY_SKELETON


class TestParseDistillOps(unittest.TestCase):
    def test_basic_ops(self):
        raw = (
            "ADD MUST: 操作前必全量阅读目标文件\n"
            "STRENGTHEN PREFER: old hint → new full rule\n"
            "WEAKEN MUST: some rule to weaken\n"
            "REMOVE MUST_NOT: deprecated rule\n"
            "NOP\n"
        )
        ops = parse_distill_ops(raw)
        self.assertEqual(len(ops), 4)
        self.assertEqual(ops[0], ("ADD", "MUST", "操作前必全量阅读目标文件"))
        self.assertEqual(ops[1], ("STRENGTHEN", "PREFER", "old hint → new full rule"))
        self.assertEqual(ops[2], ("WEAKEN", "MUST", "some rule to weaken"))
        self.assertEqual(ops[3], ("REMOVE", "MUST_NOT", "deprecated rule"))

    def test_nop_only(self):
        self.assertEqual(parse_distill_ops("NOP"), [])
        self.assertEqual(parse_distill_ops("NOP\n"), [])

    def test_empty_input(self):
        self.assertEqual(parse_distill_ops(""), [])
        self.assertEqual(parse_distill_ops("  \n  "), [])

    def test_invalid_lines_skipped(self):
        raw = "ADD MUST: valid rule\nthis is garbage\nADD PREFER: another"
        ops = parse_distill_ops(raw)
        self.assertEqual(len(ops), 2)

    def test_invalid_section_rejected(self):
        raw = "ADD INVALID_SECTION: some rule"
        ops = parse_distill_ops(raw)
        self.assertEqual(len(ops), 0)


class TestApplyOps(unittest.TestCase):
    def _skeleton(self):
        return MEMORY_SKELETON.format(date="2026-04-26", version=0)

    def test_add_to_empty(self):
        content = self._skeleton()
        ops = [("ADD", "MUST", "Always read before edit")]
        result = apply_ops(content, ops)
        self.assertIn("- Always read before edit", result)
        self.assertIn("Version: 1", result)

    def test_remove(self):
        content = self._skeleton().replace("## MUST\n", "## MUST\n- old rule to remove\n")
        ops = [("REMOVE", "MUST", "old rule to remove")]
        result = apply_ops(content, ops)
        self.assertNotIn("old rule to remove", result)

    def test_strengthen_with_arrow(self):
        content = self._skeleton().replace("## MUST\n", "## MUST\n- Evidence before claiming\n")
        ops = [("STRENGTHEN", "MUST", "Evidence before claiming → Evidence: show test output")]
        result = apply_ops(content, ops)
        self.assertIn("Evidence: show test output", result)
        self.assertNotIn("Evidence before claiming\n", result)

    def test_weaken_moves_to_prefer(self):
        content = self._skeleton().replace("## MUST\n", "## MUST\n- Strict rule here\n")
        ops = [("WEAKEN", "MUST", "Strict rule here")]
        result = apply_ops(content, ops)
        self.assertNotIn("## MUST\n- Strict rule here", result)
        self.assertIn("Strict rule here (待观察)", result)

    def test_empty_content_creates_skeleton(self):
        ops = [("ADD", "MUST", "First rule ever")]
        result = apply_ops("", ops)
        self.assertIn("## MUST", result)
        self.assertIn("- First rule ever", result)


class TestQualityGate(unittest.TestCase):
    def test_keeps_valid_bullets(self):
        obs = "- **决策模式**: 先规划再执行，引用了'谋定而后动'"
        result = quality_gate(obs)
        self.assertIn("先规划再执行", result)

    def test_rejects_insufficient_data(self):
        obs = "- 数据不足，无法构建心智模型"
        result = quality_gate(obs)
        self.assertEqual(result, "")

    def test_rejects_speculative(self):
        obs = "- 推测使用 Python 进行开发"
        result = quality_gate(obs)
        self.assertEqual(result, "")

    def test_rejects_too_short(self):
        obs = "- **技术偏好**: 用AI"
        result = quality_gate(obs)
        self.assertEqual(result, "")

    def test_preserves_headers(self):
        obs = "# Title\n- **决策模式**: 先规划再执行，使用了 codex 做验收"
        result = quality_gate(obs)
        self.assertIn("# Title", result)


class TestParseLessonEntries(unittest.TestCase):
    def test_valid_entry(self):
        raw = (
            "## bool-is-int-timestamp\n"
            "> 2026-04-18 | pk: timestamp-parse-guard | area: backend\n\n"
            "**坑**: 时间戳出现 bool\n"
            "**因**: Python bool 是 int 子类\n"
            "**法**: 解析前排除 bool\n"
        )
        entries = parse_lesson_entries(raw, "2026-04-20")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["slug"], "bool-is-int-timestamp")
        self.assertIn("2026-04-20", entries[0]["text"])  # date rewritten

    def test_missing_triple_rejected(self):
        raw = (
            "## incomplete-entry\n"
            "> 2026-04-18 | pk: test\n\n"
            "**坑**: something\n"
            "**因**: reason\n"
            # Missing **法**
        )
        entries = parse_lesson_entries(raw, "2026-04-20")
        self.assertEqual(len(entries), 0)

    def test_none_output(self):
        entries = parse_lesson_entries("NONE", "2026-04-20")
        self.assertEqual(len(entries), 0)

    def test_correction_format(self):
        raw = (
            "## naming-style-correction\n"
            "> 2026-04-26 | pk: naming-consistency | area: docs | type: correction\n\n"
            "**误**: AI 用混合命名（ai_prompts.py + ai-report.py）\n"
            "**正**: 统一用 snake_case（ai_report.py）\n"
            "**因**: 初始命名随意，后续文件沿用了不同约定\n"
        )
        entries = parse_lesson_entries(raw, "2026-04-26")
        self.assertEqual(len(entries), 1)

    def test_correction_without_cause_rejected(self):
        raw = (
            "## bad-correction\n"
            "> 2026-04-26 | pk: test | area: backend | type: correction\n\n"
            "**误**: did X\n"
            "**正**: should do Y\n"
        )
        entries = parse_lesson_entries(raw, "2026-04-26")
        self.assertEqual(len(entries), 0)

    def test_method_format(self):
        raw = (
            "## plan-before-act-method\n"
            "> 2026-04-26 | pk: plan-before-act | area: arch | type: method\n\n"
            "**法**: 谋定而后动\n"
            "**步**: 1) 探索代码库 → 2) 对齐需求 → 3) 出计划 → 4) 执行 → 5) 验证\n"
            "**用**: 非平凡任务（涉及 >3 个文件或架构变更）\n"
        )
        entries = parse_lesson_entries(raw, "2026-04-26")
        self.assertEqual(len(entries), 1)

    def test_old_format_still_works(self):
        """Entries without type: field must still parse with 坑/因/法."""
        raw = (
            "## old-lesson\n"
            "> 2026-04-18 | pk: old-pattern\n\n"
            "**坑**: something broke\n"
            "**因**: root cause\n"
            "**法**: the fix\n"
        )
        entries = parse_lesson_entries(raw, "2026-04-26")
        self.assertEqual(len(entries), 1)


class TestParseGeneYaml(unittest.TestCase):
    def test_basic_fields(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("gene_id: GEN-20260426-001\n")
            f.write("name: plan-before-act\n")
            f.write("created: 2026-04-26\n")
            f.write("decay_window_days: 90\n")
            f.write("freshness_score: 1.0\n")
            f.write("decay_status: active\n")
            path = f.name
        try:
            result = _parse_gene_yaml(Path(path))
            self.assertIsNotNone(result)
            self.assertEqual(result["gene_id"], "GEN-20260426-001")
            self.assertEqual(result["name"], "plan-before-act")
            self.assertEqual(result["decay_window_days"], "90")
        finally:
            os.unlink(path)

    def test_skips_block_scalars(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("name: test-gene\n")
            f.write("approach: |\n")
            f.write("  step 1: do something\n")
            f.write("  step 2: do more\n")
            f.write("last_used: 2026-04-20\n")
            path = f.name
        try:
            result = _parse_gene_yaml(Path(path))
            self.assertEqual(result["name"], "test-gene")
            self.assertEqual(result["last_used"], "2026-04-20")
            self.assertNotIn("approach", result)  # block scalar skipped
        finally:
            os.unlink(path)

    def test_nonexistent_file(self):
        self.assertIsNone(_parse_gene_yaml(Path("/nonexistent/gene.yaml")))

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            self.assertIsNone(_parse_gene_yaml(Path(path)))
        finally:
            os.unlink(path)


class TestTokenizeBigram(unittest.TestCase):
    def test_cjk_bigrams(self):
        tokens = _tokenize_bigram("时间戳解析")
        self.assertIn("时间", tokens)
        self.assertIn("间戳", tokens)
        self.assertIn("戳解", tokens)
        self.assertIn("解析", tokens)

    def test_latin_words(self):
        tokens = _tokenize_bigram("Python bool int")
        self.assertIn("python", tokens)
        self.assertIn("bool", tokens)
        self.assertIn("int", tokens)

    def test_mixed(self):
        tokens = _tokenize_bigram("Python的bool是int子类")
        self.assertIn("python", tokens)
        self.assertIn("bool", tokens)
        # CJK bigrams from 子类 etc
        self.assertIn("子类", tokens)

    def test_empty(self):
        self.assertEqual(_tokenize_bigram(""), set())


class TestJaccard(unittest.TestCase):
    def test_identical(self):
        s = {"a", "b", "c"}
        self.assertAlmostEqual(_jaccard(s, s), 1.0)

    def test_disjoint(self):
        self.assertAlmostEqual(_jaccard({"a"}, {"b"}), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(_jaccard({"a", "b"}, {"b", "c"}), 1/3)

    def test_empty(self):
        self.assertAlmostEqual(_jaccard(set(), {"a"}), 0.0)
        self.assertAlmostEqual(_jaccard(set(), set()), 0.0)


class TestParseAllLessonPits(unittest.TestCase):
    def test_extracts_pits(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("# LESSONS.md\n\n")
            f.write("## slug-one\n")
            f.write("<!-- absorbed: true -->\n")
            f.write("> 2026-04-18 | pk: test\n\n")
            f.write("**坑**: 时间戳出现 bool 值\n")
            f.write("**因**: reason\n**法**: fix\n\n")
            f.write("## slug-two\n")
            f.write("<!-- absorbed: false -->\n")
            f.write("> 2026-04-19 | pk: test2\n\n")
            f.write("**坑**: MCP 配置路径错误\n")
            f.write("**因**: reason\n**法**: fix\n")
            path = f.name
        try:
            pits = _parse_all_lesson_pits(Path(path))
            self.assertEqual(len(pits), 2)
            self.assertEqual(pits[0][0], "slug-one")
            self.assertIn("bool", pits[0][1])
            self.assertEqual(pits[1][0], "slug-two")
            self.assertIn("MCP", pits[1][1])
        finally:
            os.unlink(path)

    def test_nonexistent(self):
        self.assertEqual(_parse_all_lesson_pits(Path("/nonexistent")), [])


class TestCountMemoryRules(unittest.TestCase):
    def test_counts_sections(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("## MUST\n\n- rule 1\n- rule 2\n\n")
            f.write("## MUST NOT\n\n- no rule 1\n\n")
            f.write("## PREFER\n\n- prefer 1\n- prefer 2\n- prefer 3\n\n")
            f.write("## CONTEXT\n\n")
            path = f.name
        try:
            counts = _count_memory_rules(Path(path))
            self.assertEqual(counts["MUST"], 2)
            self.assertEqual(counts["MUST NOT"], 1)
            self.assertEqual(counts["PREFER"], 3)
            self.assertEqual(counts["CONTEXT"], 0)
        finally:
            os.unlink(path)


class TestExtractPatternCounts(unittest.TestCase):
    def test_counts_from_soul(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("# SOUL.md\n\n")
            f.write("\n### 2026-04-18\n")
            f.write("- observation <!-- pk: plan-before-act -->\n")
            f.write("- another <!-- pk: tight-loop -->\n")
            f.write("\n### 2026-04-19\n")
            f.write("- repeated <!-- pk: plan-before-act -->\n")
            path = f.name
        try:
            counts = extract_pattern_counts(Path(path))
            self.assertEqual(counts["plan-before-act"], 2)
            self.assertEqual(counts["tight-loop"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
