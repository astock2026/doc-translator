"""
CMC/GMP Terminology Verification for bilingual translations.

Scans translations.json against a CMC/GMP glossary, checking that:
1. Standard pharmaceutical terms are correctly translated
2. Terminology is consistent across the entire document
3. No ambiguous or non-domain translations are used for regulated terms

Produces a verification report with compliance score and flagged issues.

Usage:
    python verify_cmc.py <translations.json> [--output report.json]
"""
import json
import re
import sys
from collections import defaultdict


# ── CMC/GMP Terminology Glossary ────────────────────────────────────────
# Maps Chinese terms to their approved English equivalents.
# If a translation uses a non-approved variant, it will be flagged.

GLOSSARY = {
    # Process & Procedures
    "标准操作规程": {
        "approved": "Standard Operating Procedure",
        "aliases": ["SOP"],
        "category": "Process",
    },
    "操作规程": {
        "approved": "Operating Procedure",
        "aliases": [],
        "category": "Process",
    },

    # Equipment & Instruments
    "仪器": {
        "approved": "Instrument",
        "aliases": [],
        "category": "Equipment",
    },
    "设备": {
        "approved": "Equipment",
        "aliases": [],
        "category": "Equipment",
    },
    "仪器/设备": {
        "approved": "Instrument / Equipment",
        "aliases": [],
        "category": "Equipment",
    },

    # Calibration & Validation
    "校验": {
        "approved": "Calibration",
        "aliases": [],
        "category": "Quality",
    },
    "验证": {
        "approved": "Qualification",
        "aliases": ["Validation"],
        "note": "Use 'Qualification' for equipment/process, 'Validation' for methods/analytical",
        "category": "Quality",
    },
    "确认": {
        "approved": "Confirmation",
        "aliases": ["Verification"],
        "note": "Use 'Verification' in analytical context, 'Confirmation' for general",
        "category": "Quality",
    },

    # Document Control
    "变更控制": {
        "approved": "Change Control",
        "aliases": [],
        "category": "Document Control",
    },
    "偏差": {
        "approved": "Deviation",
        "aliases": [],
        "category": "Quality",
    },
    "纠正和预防措施": {
        "approved": "CAPA (Corrective and Preventive Action)",
        "aliases": ["CAPA", "Corrective and Preventive Action"],
        "category": "Quality",
    },
    "CAPA": {
        "approved": "CAPA",
        "aliases": [],
        "category": "Quality",
    },
    "主题专家": {
        "approved": "Subject Matter Expert",
        "aliases": ["SME"],
        "category": "Roles",
    },
    "风险评估": {
        "approved": "Risk Assessment",
        "aliases": [],
        "category": "Quality",
    },

    # Document Sections
    "使用范围": {
        "approved": "Scope",
        "aliases": [],
        "category": "Document Section",
    },
    "适用范围": {
        "approved": "Scope",
        "aliases": ["Scope of Application"],
        "category": "Document Section",
    },
    "职责": {
        "approved": "Responsibilities",
        "aliases": [],
        "category": "Document Section",
    },
    "定义": {
        "approved": "Definitions",
        "aliases": [],
        "category": "Document Section",
    },
    "缩略语": {
        "approved": "Abbreviations",
        "aliases": [],
        "category": "Document Section",
    },
    "定义/缩略语": {
        "approved": "Definitions / Abbreviations",
        "aliases": [],
        "category": "Document Section",
    },
    "参考文件": {
        "approved": "Reference Documents",
        "aliases": [],
        "category": "Document Section",
    },
    "附件": {
        "approved": "Attachments",
        "aliases": ["Appendices"],
        "category": "Document Section",
    },
    "修订历史": {
        "approved": "Revision History",
        "aliases": ["Document History"],
        "category": "Document Section",
    },

    # Personnel & Roles
    "起草人": {
        "approved": "Prepared by",
        "aliases": ["Author", "Drafted by"],
        "category": "Roles",
    },
    "审核人": {
        "approved": "Reviewed by",
        "aliases": [],
        "category": "Roles",
    },
    "批准人": {
        "approved": "Approved by",
        "aliases": [],
        "category": "Roles",
    },
    "颁发部门": {
        "approved": "Issuing Department",
        "aliases": [],
        "category": "Roles",
    },
    "生效日期": {
        "approved": "Effective Date",
        "aliases": [],
        "category": "Document Control",
    },

    # Status & Lifecycle
    "现行版本": {
        "approved": "Current Version",
        "aliases": [],
        "category": "Document Control",
    },
    "替代版本": {
        "approved": "Superseded Version",
        "aliases": [],
        "category": "Document Control",
    },
    "在域": {
        "approved": "In-Domain",
        "aliases": ["Domain"],
        "category": "Status",
    },
    "报废": {
        "approved": "Decommissioned",
        "aliases": ["Retired", "Scrapped"],
        "category": "Status",
    },

    # Manufacturing
    "生产": {
        "approved": "Manufacturing",
        "aliases": ["Production"],
        "note": "Prefer 'Manufacturing' for GMP context, 'Production' acceptable",
        "category": "Manufacturing",
    },
    "批记录": {
        "approved": "Batch Record",
        "aliases": [],
        "category": "Manufacturing",
    },
    "清洁验证": {
        "approved": "Cleaning Validation",
        "aliases": [],
        "category": "Quality",
    },
    "工艺验证": {
        "approved": "Process Validation",
        "aliases": [],
        "category": "Quality",
    },
}

# Terms that should be translated consistently everywhere
CONSISTENCY_TERMS = [
    "设备", "仪器", "校验", "验证", "偏差", "变更控制",
    "标准操作规程", "操作规程", "风险评估", "职责",
]

# Terms where translation MUST use approved form exactly
STRICT_TERMS = [
    "标准操作规程", "变更控制", "纠正和预防措施",
    "主题专家", "风险评估", "参考文件", "修订历史",
]


def extract_chinese_terms(text):
    """Extract known Chinese CMC terms found in text."""
    found = []
    for term in GLOSSARY:
        if term in text:
            found.append(term)
    return found


def check_translation_quality(chinese_text, english_translation):
    """Check if a translation uses approved CMC terminology.
    Returns list of issues found."""
    issues = []
    chn_terms = extract_chinese_terms(chinese_text)

    if not chn_terms:
        return issues

    eng_lower = english_translation.lower()

    for term in chn_terms:
        entry = GLOSSARY[term]
        approved_lower = entry["approved"].lower()

        # Check if approved term appears in translation
        approved_found = approved_lower in eng_lower

        # Check if any approved alias is used
        alias_found = False
        for alias in entry.get("aliases", []):
            if alias.lower() in eng_lower:
                alias_found = True
                break

        if not approved_found and not alias_found:
            # Check if a non-approved translation was used
            # (e.g., "check" instead of "calibration" for 校验)
            issues.append({
                "severity": "warning" if term in STRICT_TERMS else "info",
                "chinese_term": term,
                "expected": entry["approved"],
                "found_in_translation": english_translation[:80],
                "note": entry.get("note", ""),
                "category": entry["category"],
            })
        elif alias_found and term in STRICT_TERMS:
            # Alias used but strict term required
            issues.append({
                "severity": "info",
                "chinese_term": term,
                "expected": entry["approved"],
                "found_in_translation": english_translation[:80],
                "note": "Approved alias used; preferred form is: " + entry["approved"],
                "category": entry["category"],
            })

    return issues


def check_consistency(all_items):
    """Check that the same Chinese term is translated consistently."""
    term_translations = defaultdict(set)

    for item in all_items:
        chn = item.get("chinese", "")
        eng = item.get("translation", "")
        if not chn or not eng:
            continue
        for term in CONSISTENCY_TERMS:
            if term in chn:
                term_translations[term].add(eng)

    inconsistencies = []
    for term, translations in term_translations.items():
        if len(translations) > 1:
            # Same Chinese term translated differently in different places
            inconsistencies.append({
                "severity": "warning",
                "chinese_term": term,
                "expected": GLOSSARY.get(term, {}).get("approved", "N/A"),
                "found_variants": list(translations),
                "note": f"Inconsistent translation: same term '{term}' translated {len(translations)} different ways",
                "category": "Consistency",
            })

    return inconsistencies


def verify_translations(translations_path, output_path=None, content_path=None):
    """Main verification function.
    
    Args:
        translations_path: Path to translations.json
        output_path: Optional path to save report JSON
        content_path: Optional path to content.json for Chinese text reference
                      (needed when translations.json lacks 'text' fields)
    """
    with open(translations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load content.json for Chinese text if available
    content_data = None
    if content_path:
        try:
            with open(content_path, "r", encoding="utf-8") as f:
                content_data = json.load(f)
        except Exception:
            pass

    # Build content lookup maps
    para_texts = {}
    table_cell_texts = {}
    if content_data:
        for item in content_data.get("paragraphs", []):
            para_texts[item.get("index", -1)] = item.get("text", "")
        for table in content_data.get("tables", []):
            ti = table.get("index", -1)
            for row in table.get("rows", []):
                ri = row.get("index", -1)
                for cell in row.get("cells", []):
                    key = (ti, ri, cell.get("cell_index", -1), cell.get("para_index", -1))
                    table_cell_texts[key] = cell.get("text", "")

    all_issues = []
    all_items = []
    total_checked = 0
    no_chn = 0

    # Check body paragraphs
    for item in data.get("paragraphs", []):
        eng = item.get("translation", "").strip()
        chn = item.get("text", "").strip()
        idx = item.get("index", -1)
        
        # Fall back to content.json for Chinese text
        if not chn and content_data:
            chn = para_texts.get(idx, "")
        
        if not eng:
            continue
        if not chn:
            no_chn += 1
            continue
            
        total_checked += 1
        all_items.append({"chinese": chn, "translation": eng, "location": f"Paragraph {idx}"})
        issues = check_translation_quality(chn, eng)
        for issue in issues:
            issue["location"] = f"Paragraph {idx}"
        all_issues.extend(issues)

    # Check table cells
    for table in data.get("tables", []):
        ti = table.get("index", -1)
        for row in table.get("rows", []):
            ri = row.get("index", -1)
            for cell in row.get("cells", []):
                eng = cell.get("translation", "").strip()
                chn = cell.get("text", "").strip()
                ci = cell.get("cell_index", -1)
                pi = cell.get("para_index", -1)
                
                # Fall back to content.json for Chinese text
                if not chn and content_data:
                    chn = table_cell_texts.get((ti, ri, ci, pi), "")
                
                if not eng:
                    continue
                if not chn:
                    no_chn += 1
                    continue
                    
                total_checked += 1
                location = f"T{ti} R{ri} C{ci} P{pi}"
                all_items.append({"chinese": chn, "translation": eng, "location": location})
                issues = check_translation_quality(chn, eng)
                for issue in issues:
                    issue["location"] = location
                all_issues.extend(issues)

    # Consistency check
    consistency_issues = check_consistency(all_items)
    all_issues.extend(consistency_issues)

    # Score
    warnings = sum(1 for i in all_issues if i["severity"] == "warning")
    infos = sum(1 for i in all_issues if i["severity"] == "info")
    score = max(0, 100 - (warnings * 10) - (infos * 2))

    # Penalty if translations.json lacks Chinese text fields
    if no_chn > 0 and total_checked == 0:
        score = max(40, score - 30)  # Significant penalty for unverifiable translations

    report = {
        "score": min(100, score),
        "status": "PASS" if score >= 80 else ("REVIEW" if score >= 60 else "FAIL"),
        "total_checked": total_checked,
        "no_chinese_text": no_chn,
        "issues_found": len(all_issues),
        "warnings": warnings,
        "info": infos,
        "issues": all_issues,
        "summary": "",
    }

    # Generate summary
    if no_chn > 0 and total_checked == 0:
        report["summary"] = (
            f"Unable to verify — {no_chn} translations have no Chinese source text. "
            "Use the auto pipeline (LLM translate) or include 'text' fields in your JSON. "
            "Provide content.json for cross-reference."
        )
        report["status"] = "REVIEW"
    elif score >= 90:
        report["summary"] = f"Excellent — {score}/100. All CMC/GMP terminology is correct and consistent."
    elif score >= 80:
        report["summary"] = f"Good — {score}/100. Minor terminology variations detected."
    elif score >= 60:
        report["summary"] = f"Needs review — {score}/100. {warnings} terminology warnings need attention."
    else:
        report["summary"] = f"Significant issues — {score}/100. {warnings} critical terminology errors found."

    # Group issues by category
    by_category = defaultdict(list)
    for issue in all_issues:
        by_category[issue["category"]].append(issue)
    report["by_category"] = {
        cat: {"count": len(items), "issues": items}
        for cat, items in sorted(by_category.items())
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_cmc.py <translations.json> [--content content.json] [--output report.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = None
    content_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
    if "--content" in sys.argv:
        idx = sys.argv.index("--content")
        if idx + 1 < len(sys.argv):
            content_path = sys.argv[idx + 1]

    report = verify_translations(input_path, output_path, content_path)
    print(f"CMC Verification: Score {report['score']}/100 — {report['status']}")
    print(f"  Checked: {report['total_checked']} translations")
    if report.get("no_chinese_text", 0) > 0:
        print(f"  No source text: {report['no_chinese_text']}")
    print(f"  Issues: {report['issues_found']} ({report['warnings']} warnings, {report['info']} info)")
    print(f"  Summary: {report['summary']}")
    if report["issues"]:
        print("\nFlagged issues:")
        for issue in report["issues"]:
            flag = "⚠️ " if issue["severity"] == "warning" else "ℹ️ "
            print(f"  {flag}[{issue['category']}] {issue.get('location', 'unknown')}")
            print(f"      Term: '{issue['chinese_term']}' → expected '{issue['expected']}'")
            if issue.get("found_variants"):
                print(f"      Variants used: {', '.join(issue['found_variants'])}")
            print(f"      Translation: {issue.get('found_in_translation', '')[:80]}")
