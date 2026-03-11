#!/usr/bin/env python3
"""
EPC Tender PDF Intelligent Splitter  (v3 final)
=================================================
Splits large concatenated PDFs into individual documents using 10
independent heuristic detection engines — no LLM dependency.

Detection engines:
  1. Bookmark/Outline deep analysis (hierarchy, title keywords, page spans,
     clustering, auto-level-selection, label extraction)
  2. Page geometry changes (size, orientation)
  3. Page number sequence resets
  4. Text pattern matching (cover pages, section headers, TOCs)
  5. Font fingerprinting (drastic style changes)
  6. Visual density analysis (text-heavy vs drawing-heavy transitions)
  7. Blank/separator page detection
  8. Cross-page text corpus analysis (Page X of Y, running headers/footers,
     vocabulary shift, document reference tracking, boilerplate changes)
  9. Xref structure analysis (font object pools, subset prefixes, page tree
     branching, %%EOF markers from incremental concatenation)
 10. Content stream dialect detection (PDF producer fingerprinting via
     operator patterns and coordinate precision)

Each engine assigns confidence scores per page boundary. A weighted
fusion layer combines them, applies multi-engine bonuses, proximity
merging, and minimum-document-size constraints to determine final splits.

Usage:
  python splitter.py tender_package.pdf
  python splitter.py tender_package.pdf --threshold 0.6 --dry-run
  python splitter.py tender_package.pdf --report-only --report-json report.json
"""

import re
import sys
import json
import math
import hashlib
import logging
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("splitter")


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class PageFeatures:
    """Extracted features for a single page."""
    page_num: int               # 0-indexed
    width: float = 0.0
    height: float = 0.0
    orientation: str = "portrait"
    text: str = ""
    text_length: int = 0
    word_count: int = 0
    has_images: bool = False
    image_count: int = 0
    image_area_ratio: float = 0.0
    font_signature: str = ""
    dominant_fonts: list = field(default_factory=list)
    detected_page_number: Optional[int] = None
    is_blank: bool = False
    is_cover_candidate: bool = False
    is_toc_candidate: bool = False
    has_section_header: bool = False
    section_title: str = ""
    text_density: float = 0.0


@dataclass
class SplitSignal:
    """A signal suggesting a document boundary before a given page."""
    page_num: int       # boundary BEFORE this page (0-indexed)
    engine: str
    confidence: float   # 0.0 to 1.0
    reason: str = ""


@dataclass
class SplitPoint:
    """Final determined split point."""
    page_num: int
    combined_score: float
    signals: list = field(default_factory=list)
    label: str = ""


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

class FeatureExtractor:
    """Extracts per-page features from the PDF."""

    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.total_pages = len(doc)

    def extract_all(self) -> list[PageFeatures]:
        log.info(f"Extracting features from {self.total_pages} pages...")
        features = []
        for i in range(self.total_pages):
            if i % 200 == 0 and i > 0:
                log.info(f"  ...processed {i}/{self.total_pages} pages")
            features.append(self._extract_page(i))
        log.info("Feature extraction complete.")
        return features

    def _extract_page(self, idx: int) -> PageFeatures:
        page = self.doc[idx]
        rect = page.rect
        w, h = rect.width, rect.height

        pf = PageFeatures(page_num=idx)
        pf.width = round(w, 2)
        pf.height = round(h, 2)
        pf.orientation = "landscape" if w > h else "portrait"

        text = page.get_text("text") or ""
        pf.text = text
        pf.text_length = len(text.strip())
        pf.word_count = len(text.split())
        pf.is_blank = pf.word_count < 5

        area = max(w * h, 1)
        pf.text_density = pf.text_length / area

        img_list = page.get_images(full=True)
        pf.image_count = len(img_list)
        pf.has_images = pf.image_count > 0

        total_img_area = 0
        for img in img_list:
            try:
                for ir in page.get_image_rects(img[0]):
                    total_img_area += abs(ir.width * ir.height)
            except Exception:
                pass
        pf.image_area_ratio = min(total_img_area / area, 1.0) if area > 0 else 0.0

        pf.dominant_fonts, pf.font_signature = self._analyze_fonts(page)
        pf.detected_page_number = self._detect_page_number(text)
        pf.is_cover_candidate = self._is_cover_page(text, pf)
        pf.is_toc_candidate = self._is_toc_page(text)
        pf.has_section_header, pf.section_title = self._detect_section_header(text)

        return pf

    def _analyze_fonts(self, page) -> tuple[list, str]:
        font_counter = Counter()
        try:
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        fn = span.get("font", "unknown")
                        fs = round(span.get("size", 0), 1)
                        tl = len(span.get("text", "").strip())
                        if tl > 0:
                            font_counter[(fn, fs)] += tl
        except Exception:
            pass
        top = font_counter.most_common(5)
        sig_data = str(sorted([(f, s) for (f, s), _ in top]))
        sig = hashlib.md5(sig_data.encode()).hexdigest()[:12]
        return [f"{f} @ {s}pt" for (f, s), _ in top], sig

    def _detect_page_number(self, text: str) -> Optional[int]:
        lines = text.strip().split("\n")
        if not lines:
            return None
        for line in lines[-3:] + lines[:3]:
            stripped = line.strip()
            m = re.match(r'^[-\u2013\u2014\s]*(?:page\s*)?(\d{1,4})[-\u2013\u2014\s]*$',
                         stripped, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m = re.match(r'^[-\u2013\u2014\s]*((?:i{1,3}|iv|vi{0,3}|ix|xi{0,3}))[-\u2013\u2014\s]*$',
                         stripped, re.IGNORECASE)
            if m:
                return self._roman_to_int(m.group(1).lower())
        return None

    @staticmethod
    def _roman_to_int(s: str) -> int:
        vals = {'i': 1, 'v': 5, 'x': 10}
        total = 0
        for i, c in enumerate(s):
            if i + 1 < len(s) and vals.get(c, 0) < vals.get(s[i + 1], 0):
                total -= vals.get(c, 0)
            else:
                total += vals.get(c, 0)
        return total

    def _is_cover_page(self, text: str, pf: PageFeatures) -> bool:
        t = text.lower()
        # Strong patterns -- these clearly indicate a cover / title page
        strong_patterns = [
            r'(?:tender|bid)\s+(?:document|package)',
            r'(?:request\s+for\s+(?:quotation|proposal|bid))',
            r'(?:rfq|rfp|rfi|itb)\s*[:\-#]?\s*\w',
            r'(?:volume|vol\.?)\s*[\-:\s]*[ivx\d]',
            r'(?:general\s+(?:terms|conditions))',
            r'(?:scope\s+of\s+(?:work|supply))',
            r'(?:bill\s+of\s+(?:quantities|materials))',
            r'(?:technical\s+(?:specification|bid))',
            r'(?:commercial\s+(?:bid|proposal|offer))',
            r'table\s+of\s+contents',
        ]
        # Weak patterns -- common on many non-cover pages in EPC tenders;
        # only count once collectively even if multiple match
        weak_patterns = [
            r'confidential',
            r'(?:issued?\s+(?:for|date))',
            r'(?:document\s*(?:no|number|#|title))',
            r'(?:project|contract)\s*(?:no|number|#|name|title)',
        ]
        score = sum(1 for p in strong_patterns if re.search(p, t))
        has_weak = any(re.search(p, t) for p in weak_patterns)
        if has_weak:
            score += 1
        # Only consider this a cover page with substantial evidence
        # and when the page is short (actual cover pages have little body text)
        return score >= 3 and pf.word_count < 150

    def _is_toc_page(self, text: str) -> bool:
        t = text.lower()
        if re.search(r'table\s+of\s+contents|contents\s*\n|index\s*\n', t):
            return True
        dot_lines = len(re.findall(r'\.{4,}\s*\d+', text))
        total_lines = max(len(text.split('\n')), 1)
        return dot_lines / total_lines > 0.3

    def _detect_section_header(self, text: str) -> tuple[bool, str]:
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return False, ""
        # Only check the first 3 lines to avoid matching body content
        header_text = " ".join(lines[:3]).lower()
        patterns = [
            r'^(?:section|part|volume|annex|appendix|attachment|enclosure|exhibit|schedule)\s*[\-:\s]*[ivx\d]+',
            r'^(?:general|special|particular)\s+(?:terms|conditions|specifications)',
            r'^(?:scope\s+of\s+(?:work|supply))',
            r'^(?:technical\s+(?:specification|requirement|schedule))',
            r'^(?:bill\s+of\s+(?:quantities|materials))',
            r'^(?:price\s+schedule|pricing|commercial)',
            r'^(?:drawings?\s+(?:list|index|schedule))',
            # Discipline names only when followed by a document-type noun
            r'^(?:piping|instrument|electrical|civil|structural|mechanical)\s+'
            r'(?:specification|datasheet|schedule|design\s+basis|layout|philosophy)',
        ]
        for pat in patterns:
            if re.search(pat, header_text, re.IGNORECASE):
                return True, lines[0]
        return False, ""


# ─────────────────────────────────────────────
# Engine 1: Bookmark / Outline Deep Analysis
# ─────────────────────────────────────────────

class BookmarkEngine:
    """Deeply exploits PDF bookmarks / outline tree.

    Most PDF concatenation tools that preserve bookmarks simply append
    each source document's outline tree under the combined file's root.
    This engine analyzes the bookmark structure to find document-level
    boundaries through multiple sub-analyses:

      1a. Hierarchy analysis: identifies which bookmark level most likely
          represents document boundaries (not always level 1).
      1b. Title keyword matching: EPC-specific terms like Volume, Section,
          Annex, Specification, Drawing, etc.
      1c. Page span analysis: large gaps between same-level bookmarks
          suggest they mark separate documents.
      1d. Bookmark clustering: groups of child bookmarks that form a
          self-contained sub-tree likely belong to one document.
      1e. Naming pattern shifts: detects when bookmark naming conventions
          change (e.g., numbered sections to lettered appendices).
      1f. Label extraction: uses bookmark titles to generate meaningful
          names for the split documents.
    """

    NAME = "bookmark"

    # Keywords strongly associated with document-level splits
    DOCUMENT_KEYWORDS = re.compile(
        r'\b(?:volume|vol)\b\.?\s*[\-:\s]*[ivx\d]|'
        r'\b(?:part)\b\s*[\-:\s]*[ivx\d]|'
        r'\b(?:book)\b\s*[\-:\s]*[ivx\d]|'
        r'\b(?:annex|appendix|attachment|enclosure|exhibit|schedule)\b\s*[\-:\s]*[a-z\d]|'
        r'\btender\s+document\b|'
        r'\b(?:technical|commercial)\s+(?:bid|specification|proposal)\b|'
        r'\bgeneral\s+(?:terms|conditions)\b|'
        r'\bscope\s+of\s+(?:work|supply)\b|'
        r'\bbill\s+of\s+(?:quantities|materials)\b|'
        r'\bdrawing\s+(?:list|index|schedule)\b|'
        r'\b(?:price|pricing)\s+schedule\b|'
        r'\bdata\s*sheets?\b|'
        r'\bspecial\s+conditions\b|'
        r'\bparticular\s+conditions\b',
        re.IGNORECASE,
    )

    # Section-level keywords (lower confidence)
    SECTION_KEYWORDS = re.compile(
        r'\b(?:section|chapter|clause)\b\s*[\-:\s]*[\d]|'
        r'\b(?:article)\b\s*[\-:\s]*[\d]|'
        r'\btable\s+of\s+contents\b|'
        r'\bintroduction\b|'
        r'\babbreviations\b|'
        r'\bdefinitions\b',
        re.IGNORECASE,
    )

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
        if not toc:
            log.info("  BookmarkEngine: No outline/TOC found.")
            return signals

        total_entries = len(toc)
        log.info(f"  BookmarkEngine: Found {total_entries} bookmark entries.")

        # ── 1a. Hierarchy analysis ──
        # Determine which level(s) are document boundaries.
        signals.extend(self._hierarchy_analysis(toc, features))

        # ── 1b. Title keyword matching ──
        signals.extend(self._keyword_matching(toc, features))

        # ── 1c. Page span analysis ──
        signals.extend(self._page_span_analysis(toc, features))

        # ── 1d. Bookmark clustering ──
        signals.extend(self._bookmark_clustering(toc, features))

        # ── 1e. Naming pattern shifts ──
        signals.extend(self._naming_pattern_shifts(toc, features))

        # Deduplicate: keep highest confidence per page
        best_per_page = {}
        for sig in signals:
            if sig.page_num not in best_per_page or sig.confidence > best_per_page[sig.page_num].confidence:
                best_per_page[sig.page_num] = sig
        signals = list(best_per_page.values())

        log.info(f"  BookmarkEngine: {len(signals)} bookmark signals (from {total_entries} entries).")
        return signals

    def _hierarchy_analysis(self, toc, features) -> list[SplitSignal]:
        """Analyze bookmark levels to decide which ones are document boundaries.

        When multiple PDFs are concatenated and their bookmarks preserved,
        the typical pattern is:
        - All source docs' top-level bookmarks appear at the same level
        - Under each top-level entry, child bookmarks form a subtree

        Strategy: find the level where entries have the largest average
        page span between them. That's likely the document level.
        """
        signals = []
        n_pages = len(features)

        # Group entries by level
        by_level = defaultdict(list)
        for level, title, page_num in toc:
            by_level[level].append((title, page_num))

        if not by_level:
            return signals

        # For each level, compute average page span between consecutive entries
        level_stats = {}
        for level, entries in by_level.items():
            if len(entries) < 2:
                continue
            pages = sorted(set(e[1] for e in entries))
            if len(pages) < 2:
                continue
            spans = [pages[i + 1] - pages[i] for i in range(len(pages) - 1)]
            avg_span = statistics.mean(spans)
            min_span = min(spans)
            level_stats[level] = {
                'count': len(entries),
                'avg_span': avg_span,
                'min_span': min_span,
                'pages': pages,
                'entries': entries,
            }

        if not level_stats:
            return signals

        # The document-boundary level typically has:
        # - Relatively few entries (matching number of source documents)
        # - Large average page span
        # Score each level
        best_level = None
        best_score = 0
        for level, stats in level_stats.items():
            # Score: reward large spans, penalize too many entries
            # A level with 3-20 entries and large spans is ideal
            count = stats['count']
            avg_span = stats['avg_span']
            if count < 2 or count > n_pages / 2:
                continue
            score = avg_span * math.log2(max(count, 2))
            if 2 <= count <= 30:
                score *= 1.5  # sweet spot for document count
            if score > best_score:
                best_score = score
                best_level = level

        if best_level is not None and best_level in level_stats:
            entries = level_stats[best_level]['entries']
            avg_span = level_stats[best_level]['avg_span']
            for title, page_num in entries:
                if page_num <= 1:
                    continue
                idx = page_num - 1
                if 0 < idx < n_pages:
                    # Higher confidence if this is the shallowest level
                    min_level = min(by_level.keys())
                    conf = 0.9 if best_level == min_level else 0.7
                    signals.append(SplitSignal(
                        page_num=idx, engine=self.NAME, confidence=conf,
                        reason=f"Hierarchy analysis: level-{best_level} bookmark '{title}' "
                               f"(avg span={avg_span:.0f} pages, best document-level)"
                    ))

        return signals

    def _keyword_matching(self, toc, features) -> list[SplitSignal]:
        """Match bookmark titles against EPC-specific keywords."""
        signals = []
        n_pages = len(features)

        for level, title, page_num in toc:
            if page_num <= 1:
                continue
            idx = page_num - 1
            if idx <= 0 or idx >= n_pages:
                continue

            # Document-level keywords
            if self.DOCUMENT_KEYWORDS.search(title):
                conf = 0.85 if level <= 2 else 0.6
                signals.append(SplitSignal(
                    page_num=idx, engine=self.NAME, confidence=conf,
                    reason=f"Keyword match (document-level): '{title}' [level {level}]"
                ))
            # Section-level keywords (weaker signal)
            elif self.SECTION_KEYWORDS.search(title) and level <= 2:
                signals.append(SplitSignal(
                    page_num=idx, engine=self.NAME, confidence=0.35,
                    reason=f"Keyword match (section-level): '{title}' [level {level}]"
                ))

        return signals

    def _page_span_analysis(self, toc, features) -> list[SplitSignal]:
        """Look for unusually large gaps between same-level bookmarks.

        If most level-1 bookmarks are 5-10 pages apart but one gap is 200
        pages, that 200-page chunk is probably a separate document within
        a larger section structure. Similarly, if two level-1 bookmarks
        point to the same page (or adjacent pages), they're likely both
        from the same document's internal structure.
        """
        signals = []
        n_pages = len(features)

        by_level = defaultdict(list)
        for level, title, page_num in toc:
            by_level[level].append((title, page_num))

        for level in [1, 2]:
            entries = by_level.get(level, [])
            if len(entries) < 3:
                continue

            pages = [e[1] for e in entries]
            spans = []
            for i in range(1, len(pages)):
                spans.append((i, pages[i] - pages[i - 1], entries[i]))

            if len(spans) < 2:
                continue

            span_values = [s[1] for s in spans]
            if not span_values:
                continue

            mean_span = statistics.mean(span_values)
            if len(span_values) > 2:
                std_span = statistics.stdev(span_values)
            else:
                std_span = 0

            # Flag entries where the span to the PREVIOUS entry is very large
            # (suggests the previous entry was the last bookmark of a different doc)
            for i_pos, span, (title, page_num) in spans:
                if std_span > 0 and span > mean_span + 1.5 * std_span and span > 20:
                    idx = page_num - 1
                    if 0 < idx < n_pages:
                        z = (span - mean_span) / std_span if std_span > 0 else 0
                        conf = min(0.4 + z * 0.1, 0.8)
                        signals.append(SplitSignal(
                            page_num=idx, engine=self.NAME, confidence=conf,
                            reason=f"Large page span before level-{level} bookmark: "
                                   f"'{title}' (span={span} pages, mean={mean_span:.0f})"
                        ))

        return signals

    def _bookmark_clustering(self, toc, features) -> list[SplitSignal]:
        """Identify clusters of bookmarks forming self-contained subtrees.

        When PDFs are concatenated with bookmarks, each source doc's
        bookmarks form a contiguous cluster. This detects where one
        cluster ends and another begins by looking at child bookmark
        density and parent-child groupings.
        """
        signals = []
        n_pages = len(features)
        if len(toc) < 4:
            return signals

        # Build parent-children mapping based on outline levels
        # Each level-1 entry "owns" subsequent entries until the next level-1
        top_level = min(entry[0] for entry in toc)
        clusters = []  # [(title, start_page, end_page, num_children), ...]

        current_cluster = None
        for level, title, page_num in toc:
            if level == top_level:
                if current_cluster is not None:
                    clusters.append(current_cluster)
                current_cluster = {
                    'title': title,
                    'page': page_num,
                    'children': 0,
                    'child_pages': [],
                }
            elif current_cluster is not None:
                current_cluster['children'] += 1
                current_cluster['child_pages'].append(page_num)

        if current_cluster is not None:
            clusters.append(current_cluster)

        if len(clusters) < 2:
            return signals

        # Each cluster corresponds to a top-level bookmark and its children.
        # The page range of a cluster spans from its start page to just before
        # the next cluster's start page.
        for i in range(1, len(clusters)):
            cl = clusters[i]
            idx = cl['page'] - 1
            if 0 < idx < n_pages:
                prev_cl = clusters[i - 1]
                # Stronger signal if both clusters have children
                # (i.e., they're real document trees, not orphan bookmarks)
                has_children = prev_cl['children'] > 0 and cl['children'] > 0
                conf = 0.8 if has_children else 0.55
                signals.append(SplitSignal(
                    page_num=idx, engine=self.NAME, confidence=conf,
                    reason=f"Bookmark cluster boundary: '{prev_cl['title']}' "
                           f"({prev_cl['children']} children) -> "
                           f"'{cl['title']}' ({cl['children']} children)"
                ))

        return signals

    def _naming_pattern_shifts(self, toc, features) -> list[SplitSignal]:
        """Detect when bookmark naming conventions change abruptly.

        Source documents from different companies or departments often
        use different naming conventions. E.g., one uses "1.0 Scope",
        "2.0 Design" while another uses "Article I", "Article II".
        A shift in the numbering or naming scheme is a boundary clue.
        """
        signals = []
        n_pages = len(features)

        # Focus on a consistent level
        top_level = min(entry[0] for entry in toc) if toc else 1
        entries = [(title, page_num) for level, title, page_num in toc
                   if level <= top_level + 1]

        if len(entries) < 4:
            return signals

        # Classify each title's numbering scheme
        def classify_scheme(title):
            t = title.strip()
            if re.match(r'^\d+[\.\d]*\s', t):
                return 'decimal'       # "1.0 Scope", "2.3 Requirements"
            if re.match(r'^[A-Z]\.\s', t):
                return 'alpha_upper'   # "A. General", "B. Specific"
            if re.match(r'^[a-z]\.\s', t):
                return 'alpha_lower'
            if re.match(r'^(?:Article|Section|Chapter)\s+[IVXLC]+', t, re.IGNORECASE):
                return 'roman'
            if re.match(r'^(?:Article|Section|Chapter)\s+\d', t, re.IGNORECASE):
                return 'named_decimal'
            if re.match(r'^(?:Volume|Vol|Part|Book)\s', t, re.IGNORECASE):
                return 'volume'
            if re.match(r'^(?:Annex|Appendix|Attachment|Schedule)\s', t, re.IGNORECASE):
                return 'appendix'
            return 'freeform'

        schemes = [(classify_scheme(t), t, p) for t, p in entries]

        # Find transitions between different schemes
        run_start = 0
        for i in range(1, len(schemes)):
            curr_scheme = schemes[i][0]
            prev_scheme = schemes[i - 1][0]

            if curr_scheme != prev_scheme and curr_scheme != 'freeform' and prev_scheme != 'freeform':
                run_len = i - run_start
                if run_len >= 2:
                    page_num = schemes[i][2]
                    idx = page_num - 1
                    if 0 < idx < n_pages:
                        signals.append(SplitSignal(
                            page_num=idx, engine=self.NAME, confidence=0.5,
                            reason=f"Naming pattern shift: {prev_scheme} -> {curr_scheme} "
                                   f"at '{schemes[i][1]}' (after {run_len} '{prev_scheme}' entries)"
                        ))
                run_start = i

        return signals


# ─────────────────────────────────────────────
# Engine 2: Page Geometry
# ─────────────────────────────────────────────

class GeometryEngine:
    NAME = "geometry"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        for i in range(1, len(features)):
            prev, curr = features[i - 1], features[i]
            size_changed = (abs(prev.width - curr.width) > 2 or
                            abs(prev.height - curr.height) > 2)
            orient_changed = prev.orientation != curr.orientation

            if size_changed and orient_changed:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.6,
                    reason=f"Size AND orientation change: "
                           f"{prev.width}x{prev.height} {prev.orientation} -> "
                           f"{curr.width}x{curr.height} {curr.orientation}"
                ))
            elif size_changed:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.5,
                    reason=f"Page size change: {prev.width}x{prev.height} -> "
                           f"{curr.width}x{curr.height}"
                ))
            elif orient_changed:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.15,
                    reason=f"Orientation change: {prev.orientation} -> {curr.orientation}"
                ))

        log.info(f"  GeometryEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 3: Page Number Resets
# ─────────────────────────────────────────────

class PageNumberEngine:
    NAME = "page_number"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        nums = [(f.page_num, f.detected_page_number) for f in features
                if f.detected_page_number is not None]

        if len(nums) < 5:
            log.info("  PageNumberEngine: Too few page numbers detected, skipping.")
            return signals

        for i in range(1, len(nums)):
            pg_idx, pg_num = nums[i]
            prev_pg_idx, prev_pg_num = nums[i - 1]

            if pg_num <= 2 and prev_pg_num > 3:
                gap = pg_idx - prev_pg_idx
                conf = 0.85 if gap <= 2 else 0.6
                signals.append(SplitSignal(
                    page_num=pg_idx, engine=self.NAME, confidence=conf,
                    reason=f"Page number reset: {prev_pg_num} -> {pg_num} "
                           f"(pdf pages {prev_pg_idx + 1} -> {pg_idx + 1})"
                ))
            elif pg_num < prev_pg_num - 5:
                signals.append(SplitSignal(
                    page_num=pg_idx, engine=self.NAME, confidence=0.5,
                    reason=f"Page number jump backward: {prev_pg_num} -> {pg_num}"
                ))

        log.info(f"  PageNumberEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 4: Text Pattern Matching (per-page)
# ─────────────────────────────────────────────

class TextPatternEngine:
    NAME = "text_pattern"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        for i, feat in enumerate(features):
            if i == 0:
                continue

            if feat.is_cover_candidate:
                prev = features[i - 1]
                conf = 0.75 if prev.is_blank else 0.6
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=conf,
                    reason="Cover/title page detected"
                ))

            if feat.is_toc_candidate and not features[i - 1].is_toc_candidate:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.65,
                    reason="Table of contents page start"
                ))

            if feat.has_section_header and feat.word_count < 60:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.55,
                    reason=f"Section header page: '{feat.section_title}'"
                ))

            # Document number change in header area
            self._check_doc_number_change(features, i, signals)

        log.info(f"  TextPatternEngine: {len(signals)} signals.")
        return signals

    def _check_doc_number_change(self, features, i, signals):
        if i == 0:
            return
        doc_num_pat = r'(?:doc(?:ument)?\.?\s*(?:no|#|number)\.?\s*[:\-]?\s*)([\w\-/]+)'
        curr_text = features[i].text[:500]
        prev_text = features[i - 1].text[:500]
        curr_m = re.search(doc_num_pat, curr_text, re.IGNORECASE)
        prev_m = re.search(doc_num_pat, prev_text, re.IGNORECASE)
        if curr_m and prev_m and curr_m.group(1) != prev_m.group(1):
            signals.append(SplitSignal(
                page_num=i, engine=self.NAME, confidence=0.7,
                reason=f"Document number change: {prev_m.group(1)} -> {curr_m.group(1)}"
            ))


# ─────────────────────────────────────────────
# Engine 5: Font Fingerprinting
# ─────────────────────────────────────────────

class FontFingerprintEngine:
    NAME = "font_fingerprint"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        run_length = 0
        for i in range(1, len(features)):
            if features[i - 1].font_signature == features[i].font_signature:
                run_length += 1
            else:
                if run_length >= 3:
                    conf = min(0.3 + (run_length * 0.02), 0.6)
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=conf,
                        reason=f"Font signature change after {run_length + 1} consistent pages"
                    ))
                run_length = 0
        log.info(f"  FontFingerprintEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 6: Visual Density
# ─────────────────────────────────────────────

class DensityEngine:
    NAME = "density"
    WINDOW = 5

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        n = len(features)
        if n < self.WINDOW * 2 + 1:
            return signals

        densities = [f.text_density for f in features]
        img_ratios = [f.image_area_ratio for f in features]

        for i in range(self.WINDOW, n - self.WINDOW):
            bd = statistics.mean(densities[i - self.WINDOW:i])
            ad = statistics.mean(densities[i:i + self.WINDOW])
            bi = statistics.mean(img_ratios[i - self.WINDOW:i])
            ai = statistics.mean(img_ratios[i:i + self.WINDOW])

            if bd > 0 and ad > 0:
                ratio = max(bd, ad) / min(bd, ad)
                if ratio > 3.0:
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=0.35,
                        reason=f"Text density shift: {bd:.4f} -> {ad:.4f}"
                    ))
            if abs(ai - bi) > 0.3:
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.3,
                    reason=f"Image ratio shift: {bi:.2f} -> {ai:.2f}"
                ))

        log.info(f"  DensityEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 7: Blank / Separator Pages
# ─────────────────────────────────────────────

class BlankPageEngine:
    NAME = "blank_separator"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        for i in range(1, len(features) - 1):
            curr = features[i]
            nxt = features[i + 1]
            if curr.is_blank and not nxt.is_blank:
                signals.append(SplitSignal(
                    page_num=i + 1, engine=self.NAME, confidence=0.4,
                    reason=f"Blank separator page at pdf page {i + 1}"
                ))
            if "intentionally" in curr.text.lower() and "blank" in curr.text.lower():
                signals.append(SplitSignal(
                    page_num=i + 1, engine=self.NAME, confidence=0.55,
                    reason=f"'Intentionally left blank' page at pdf page {i + 1}"
                ))
        log.info(f"  BlankPageEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 8: Cross-Page Text Corpus Analysis
# ─────────────────────────────────────────────

class TextCorpusEngine:
    """Cross-page text analysis: Page X of Y, running headers/footers,
    vocabulary shift, document reference tracking, boilerplate changes."""

    NAME = "text_corpus"

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        texts = [f.text for f in features]

        s1 = self._page_x_of_y(features)
        s2 = self._running_header_footer(texts)
        s3 = self._vocabulary_shift(texts)
        s4 = self._document_ref_tracking(texts)
        s5 = self._boilerplate_change(texts)

        signals = s1 + s2 + s3 + s4 + s5
        log.info(f"  TextCorpusEngine: {len(signals)} signals "
                 f"(XofY={len(s1)}, hdr/ftr={len(s2)}, vocab={len(s3)}, "
                 f"docref={len(s4)}, boilerplate={len(s5)})")
        return signals

    def _page_x_of_y(self, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        pxy_pat = re.compile(
            r'(?:page|sheet|pg\.?|p\.?)?\s*(\d{1,4})\s*(?:of|/)\s*(\d{1,4})',
            re.IGNORECASE)

        entries = []
        for feat in features:
            lines = feat.text.strip().split('\n')
            search = '\n'.join(lines[:5] + lines[-5:])
            matches = pxy_pat.findall(search)
            if matches:
                for xs, ys in matches:
                    x, y = int(xs), int(ys)
                    if 1 <= x <= y <= 5000:
                        entries.append((feat.page_num, x, y))
                        break

        if len(entries) < 3:
            return signals

        for i in range(1, len(entries)):
            pg_idx, x, y = entries[i]
            ppg, px, py = entries[i - 1]

            if y != py:
                was_near_end = (py - px) <= 2
                conf = 0.9 if was_near_end else 0.75
                signals.append(SplitSignal(
                    page_num=pg_idx, engine=self.NAME, confidence=conf,
                    reason=f"'Page X of Y' total changed: of {py} (at pg {px}) -> of {y} (at pg {x})"
                ))
            elif x <= 1 and px > 1:
                signals.append(SplitSignal(
                    page_num=pg_idx, engine=self.NAME, confidence=0.85,
                    reason=f"'Page X of Y' counter reset: {px}/{py} -> {x}/{y}"
                ))

        # Predict from last page (x == y)
        existing = {s.page_num for s in signals}
        for i in range(len(entries) - 1):
            pg_idx, x, y = entries[i]
            nxt_pg = entries[i + 1][0]
            if x == y and (nxt_pg - pg_idx) <= 2:
                target = pg_idx + 1
                if target not in existing and target < len(features):
                    signals.append(SplitSignal(
                        page_num=target, engine=self.NAME, confidence=0.7,
                        reason=f"Previous page was final page ({x} of {y})"
                    ))

        return signals

    def _running_header_footer(self, texts: list[str]) -> list[SplitSignal]:
        signals = []
        n = len(texts)
        if n < 6:
            return signals

        def extract_hf(text):
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            h = ' | '.join(lines[:2]) if len(lines) >= 2 else (lines[0] if lines else '')
            f = ' | '.join(lines[-2:]) if len(lines) >= 2 else (lines[-1] if lines else '')
            return h, f

        def normalize(s):
            s = re.sub(r'\b\d{1,4}\b', '#', s)
            return re.sub(r'\s+', ' ', s).strip().lower()

        raw = [extract_hf(t) for t in texts]
        norm_h = [normalize(r[0]) for r in raw]
        norm_f = [normalize(r[1]) for r in raw]

        def find_changes(items, label):
            sigs = []
            run_start = 0
            for i in range(1, len(items)):
                if items[i] != items[i - 1] or not items[i]:
                    run_len = i - run_start
                    if run_len >= 3 and items[i] and items[i - 1]:
                        pw = set(items[i - 1].split())
                        cw = set(items[i].split())
                        union = pw | cw
                        inter = pw & cw
                        jacc = len(inter) / max(len(union), 1)
                        if jacc < 0.5:
                            conf = min(0.5 + (run_len * 0.03), 0.85)
                            sigs.append(SplitSignal(
                                page_num=i, engine=self.NAME, confidence=conf,
                                reason=f"Running {label} changed after {run_len} pages "
                                       f"(overlap: {jacc:.0%})"
                            ))
                    run_start = i
            return sigs

        signals.extend(find_changes(norm_h, "header"))
        signals.extend(find_changes(norm_f, "footer"))
        return signals

    def _vocabulary_shift(self, texts, window=8, step=4) -> list[SplitSignal]:
        signals = []
        n = len(texts)
        if n < window * 2 + step:
            return signals

        stop = {'the','a','an','and','or','but','in','on','at','to','for','of',
                'with','by','from','is','are','was','were','be','been','being',
                'have','has','had','do','does','did','will','would','could',
                'should','may','might','shall','this','that','these','those',
                'it','its','not','no','as','if','than','so','such','all','any',
                'each','per','also','which','who','whom','what','when','where'}

        def tok(text):
            return [w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in stop]

        page_tok = [tok(t) for t in texts]

        def wtf(s, e):
            c = Counter()
            for i in range(s, min(e, n)):
                c.update(page_tok[i])
            return c

        def cosim(a, b):
            terms = set(a) | set(b)
            if not terms: return 1.0
            dot = sum(a.get(t, 0) * b.get(t, 0) for t in terms)
            m1 = math.sqrt(sum(v*v for v in a.values()))
            m2 = math.sqrt(sum(v*v for v in b.values()))
            return dot / (m1 * m2) if m1 and m2 else 0.0

        sims, pos = [], []
        for i in range(window, n - window, step):
            sims.append(cosim(wtf(i - window, i), wtf(i, i + window)))
            pos.append(i)

        if len(sims) < 3:
            return signals

        mn = statistics.mean(sims)
        sd = statistics.stdev(sims) if len(sims) > 1 else 0

        for j in range(1, len(sims) - 1):
            s = sims[j]
            if s < sims[j-1] and s < sims[j+1]:
                z = (mn - s) / sd if sd > 0 else 0
                if z > 1.0:
                    signals.append(SplitSignal(
                        page_num=pos[j], engine=self.NAME,
                        confidence=min(0.3 + z * 0.15, 0.7),
                        reason=f"Vocabulary shift: cosine={s:.3f} (mean={mn:.3f}, z={z:.1f})"
                    ))
        return signals

    def _document_ref_tracking(self, texts) -> list[SplitSignal]:
        signals = []
        ref_pats = [
            re.compile(r'(?:doc(?:ument)?\.?\s*(?:no|#|number|ref)\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
            re.compile(r'(?:ref(?:erence)?\.?\s*(?:no|#)?\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
            re.compile(r'(?:spec(?:ification)?\.?\s*(?:no|#)?\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
            re.compile(r'(?:contract\s*(?:no|#)\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
            re.compile(r'(?:job\s*(?:no|#)\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
            re.compile(r'(?:drawing\s*(?:no|#)\.?\s*[:\-]?\s*)([\w\-/.:]+)', re.I),
        ]

        def extract(text):
            lines = text.strip().split('\n')
            area = '\n'.join(lines[:4] + lines[-4:])
            refs = {}
            for p in ref_pats:
                m = p.search(area)
                if m:
                    refs[p.pattern[:30]] = m.group(1).strip().rstrip('.')
            return refs

        page_refs = [extract(t) for t in texts]
        all_keys = set()
        for r in page_refs:
            all_keys |= set(r.keys())

        for key in all_keys:
            seq = [(i, page_refs[i][key]) for i in range(len(page_refs)) if key in page_refs[i]]
            if len(seq) < 4:
                continue
            run_start = 0
            for j in range(1, len(seq)):
                pi, v = seq[j]
                ppi, pv = seq[j - 1]
                if v != pv:
                    run_len = j - run_start
                    if run_len >= 3:
                        bp = pi if (pi - ppi) <= 2 else ppi + 1
                        signals.append(SplitSignal(
                            page_num=bp, engine=self.NAME,
                            confidence=min(0.6 + run_len * 0.03, 0.85),
                            reason=f"Doc reference changed: '{pv}' ({run_len} pages) -> '{v}'"
                        ))
                    run_start = j
        return signals

    def _boilerplate_change(self, texts) -> list[SplitSignal]:
        signals = []
        n = len(texts)
        if n < 6:
            return signals

        bp_pats = [
            (re.compile(r'(?:prepared\s+by|issued\s+by|contractor|client|owner|vendor)\s*[:\-]?\s*(.{5,50})', re.I), 'entity'),
            (re.compile(r'((?:confidential|proprietary|restricted|internal\s+use).{0,80})', re.I), 'confidentiality'),
            (re.compile(r'(?:rev(?:ision)?\.?\s*[:\-]?\s*)([a-z0-9]{1,5})', re.I), 'revision'),
            (re.compile(r'(?:project\s*(?:no|#|name)?\.?\s*[:\-]?\s*)([\w\-/\s]{3,40})', re.I), 'project'),
        ]

        def extract(text):
            lines = text.strip().split('\n')
            area = '\n'.join(lines[:5] + lines[-5:])
            bp = {}
            for p, lbl in bp_pats:
                m = p.search(area)
                if m:
                    bp[lbl] = m.group(1).strip().lower()
            return bp

        page_bp = [extract(t) for t in texts]
        all_lbls = set()
        for b in page_bp:
            all_lbls |= set(b.keys())

        for lbl in all_lbls:
            seq = [(i, page_bp[i][lbl]) for i in range(n) if lbl in page_bp[i]]
            if len(seq) < 4:
                continue
            rs = 0
            for j in range(1, len(seq)):
                pi, v = seq[j]
                ppi, pv = seq[j - 1]
                if v != pv:
                    rl = j - rs
                    if rl >= 3:
                        bp = pi if (pi - ppi) <= 2 else ppi + 1
                        signals.append(SplitSignal(
                            page_num=bp, engine=self.NAME,
                            confidence=min(0.45 + rl * 0.03, 0.7),
                            reason=f"Boilerplate '{lbl}' changed after {rl} pages: "
                                   f"'{pv[:30]}' -> '{v[:30]}'"
                        ))
                    rs = j
        return signals


# ─────────────────────────────────────────────
# Engine 9: Xref Structure Analysis
# ─────────────────────────────────────────────

class XrefStructureEngine:
    """Analyzes internal PDF object structure: font/image xref pools,
    subset prefixes, %%EOF markers, page tree branching."""

    NAME = "xref_structure"

    # Minimum consecutive pages with the same font pool before a change
    # is considered meaningful (avoids noise from isolated drawing pages).
    MIN_FONT_RUN = 3

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        n = len(doc)

        # 1. Font object reference clustering (with run-length gating)
        page_font_xrefs = []
        for i in range(n):
            xrefs = set()
            try:
                for f in doc[i].get_fonts(full=True):
                    xrefs.add(f[0])
            except Exception:
                pass
            page_font_xrefs.append(xrefs)

        # Compute per-page Jaccard with previous page
        jaccards = [None]  # index 0 has no predecessor
        for i in range(1, n):
            prev_s, curr_s = page_font_xrefs[i - 1], page_font_xrefs[i]
            if not prev_s or not curr_s:
                jaccards.append(None)
                continue
            inter = prev_s & curr_s
            union = prev_s | curr_s
            jaccards.append(len(inter) / len(union) if union else 1.0)

        # Only emit a signal after a stable run of similar font pools
        run_len = 0
        for i in range(1, n):
            j = jaccards[i]
            if j is None:
                run_len = 0
                continue
            if j > 0.5:
                run_len += 1
            else:
                if run_len >= self.MIN_FONT_RUN:
                    conf = 0.55 if j == 0.0 else 0.35
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=conf,
                        reason=f"Font xref pool changed (Jaccard={j:.2f}) "
                               f"after {run_len} stable pages"
                    ))
                run_len = 0

        # 2. Image/XObject reference clustering
        page_img_xrefs = []
        for i in range(n):
            xrefs = set()
            try:
                for img in doc[i].get_images(full=True):
                    xrefs.add(img[0])
            except Exception:
                pass
            page_img_xrefs.append(xrefs)

        window = 5
        for i in range(window, n):
            recent = set()
            for j in range(max(0, i - window), i):
                recent |= page_img_xrefs[j]
            curr = page_img_xrefs[i]
            if recent and curr and len(recent) >= 2 and not (recent & curr):
                signals.append(SplitSignal(
                    page_num=i, engine=self.NAME, confidence=0.3,
                    reason=f"Image xref pool reset (no overlap with preceding {window} pages)"
                ))

        # 3. Font subset prefix changes (require majority changed + run gating)
        page_subsets = []
        for i in range(n):
            prefixes = {}
            try:
                for f in doc[i].get_fonts(full=True):
                    name = f[3] if len(f) > 3 else ""
                    m = re.match(r'^([A-Z]{6})\+(.+)$', name)
                    if m:
                        prefixes[m.group(2)] = m.group(1)
            except Exception:
                pass
            page_subsets.append(prefixes)

        subset_run = 0
        for i in range(1, n):
            common = set(page_subsets[i - 1]) & set(page_subsets[i])
            if not common:
                subset_run += 1
                continue
            changed = [b for b in common if page_subsets[i - 1][b] != page_subsets[i][b]]
            change_ratio = len(changed) / len(common)
            if change_ratio < 0.6:
                subset_run += 1
            else:
                if subset_run >= self.MIN_FONT_RUN:
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=0.4,
                        reason=f"Font subset prefixes changed ({len(changed)}/{len(common)} fonts) "
                               f"after {subset_run} stable pages"
                    ))
                subset_run = 0

        # 4. %%EOF markers (incremental concatenation)
        try:
            raw_path = doc.name
            if raw_path:
                with open(raw_path, 'rb') as f:
                    raw = f.read()
                eofs = [m.start() for m in re.finditer(rb'%%EOF', raw)]
                if len(eofs) > 1:
                    log.info(f"  XrefStructureEngine: {len(eofs)} %%EOF markers found.")
                    for pos in eofs[:-1]:
                        chunk = raw[pos:pos + 200]
                        if re.search(rb'startxref\s+\d+', chunk):
                            frac = pos / len(raw)
                            approx = max(1, min(int(frac * n), n - 1))
                            signals.append(SplitSignal(
                                page_num=approx, engine=self.NAME, confidence=0.7,
                                reason=f"%%EOF at byte {pos} (~page {approx + 1})"
                            ))
        except Exception as e:
            log.debug(f"  XrefStructureEngine: raw byte scan failed: {e}")

        # 5. Page tree branching -- only useful when concatenation merges
        # page trees at a high level.  Balanced page-tree splits are normal
        # PDF implementation details, so we only fire when the number of
        # branching points is small relative to total pages (i.e., the tree
        # was clearly stitched from a few source documents).
        try:
            parents = []
            for i in range(n):
                xref = doc[i].xref
                parents.append(doc.xref_get_key(xref, "Parent"))
            changes = [i for i in range(1, n) if parents[i] != parents[i - 1]]
            # Only useful as a signal when there are very few branch points
            # (roughly matching the number of source documents, not 100s).
            if 1 <= len(changes) <= max(30, n // 100):
                for i in changes:
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=0.5,
                        reason=f"Page tree parent changed ({len(changes)} total branch points)"
                    ))
        except Exception as e:
            log.debug(f"  XrefStructureEngine: page tree scan failed: {e}")

        log.info(f"  XrefStructureEngine: {len(signals)} signals.")
        return signals


# ─────────────────────────────────────────────
# Engine 10: Content Stream Dialect
# ─────────────────────────────────────────────

class ContentStreamDialectEngine:
    """Detects PDF producer fingerprint changes via content stream
    operator patterns and coordinate precision."""

    NAME = "content_dialect"

    # Pre-compiled feature detection patterns
    _FEAT_PATTERNS = [
        (re.compile(rb'%%'), 'comments'),
        (re.compile(rb'\bDo\b'), 'xobject_do'),
        (re.compile(rb'\bre\b'), 'rect_op'),
        (re.compile(rb'\bBI\b'), 'inline_img'),  # just check presence, no .*?
        (re.compile(rb'\bBMC\b'), 'marked_content'),
        (re.compile(rb'\bBDC\b'), 'marked_dict'),
        (re.compile(rb'\bgs\b'), 'gstate'),
        (re.compile(rb'\bcs\b'), 'colorspace'),
        (re.compile(rb'\bSCN\b'), 'scn_color'),
        (re.compile(rb'\bTf\b'), 'font_op'),
        (re.compile(rb'\bTm\b'), 'text_matrix'),
        (re.compile(rb'\bTd\b'), 'text_pos'),
        (re.compile(rb'\bTJ\b'), 'array_show'),
        (re.compile(rb'\bTj\b'), 'simple_show'),
        (re.compile(rb'\bcm\b'), 'concat_matrix'),
        (re.compile(rb'\bW\b'), 'clip'),
        (re.compile(rb'\bW\*'), 'eo_clip'),
    ]
    _RE_COORDS = re.compile(rb'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(?:Td|Tm|m|l|c|re)')
    _RE_T_OPS = re.compile(rb'\b(?:Tj|TJ|Td|Tm|Tf)\b')
    _RE_G_OPS = re.compile(rb'\b(?:m|l|c|re|h|f|S|B|W)\b')
    _RE_PREC_COORDS = re.compile(rb'(\d+\.?\d*)\s+\d+\.?\d*\s+(?:Td|Tm|m|l)')

    def run(self, doc: fitz.Document, features: list[PageFeatures]) -> list[SplitSignal]:
        signals = []
        n = len(doc)

        # Read all content streams once and cache (truncate to 32KB for
        # fingerprinting -- the first portion is sufficient to characterise
        # the producer's style and avoids slow regex on huge drawing streams).
        MAX_STREAM = 32768
        streams = []
        for i in range(n):
            try:
                s = doc[i].read_contents()
                streams.append(s[:MAX_STREAM] if s else None)
            except Exception:
                streams.append(None)

        # Build fingerprints from cached streams
        fps = [self._fp(s) for s in streams]

        run_len = 0
        for i in range(1, len(fps)):
            p, c = fps[i - 1], fps[i]
            if not p or not c:
                run_len = 0
                continue
            keys = set(p) | set(c)
            diffs = sum(1 for k in keys if p.get(k) != c.get(k))
            dr = diffs / max(len(keys), 1)
            if dr < 0.15:
                run_len += 1
            else:
                if run_len >= 3 and dr > 0.35:
                    conf = min(0.3 + dr * 0.5 + run_len * 0.02, 0.75)
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=conf,
                        reason=f"Content stream dialect shift: {diffs}/{len(keys)} features "
                               f"after {run_len + 1} consistent pages"
                    ))
                run_len = 0

        # Precision shifts using the same cached streams
        signals.extend(self._precision_shifts(streams))
        log.info(f"  ContentStreamDialectEngine: {len(signals)} signals.")
        return signals

    def _fp(self, stream: bytes | None) -> dict:
        if not stream:
            return {}
        fp = {}
        for pat, name in self._FEAT_PATTERNS:
            fp[name] = bool(pat.search(stream))
        coords = self._RE_COORDS.findall(stream)
        if coords:
            precs = []
            for x, y in coords[:50]:
                for v in (x, y):
                    vs = v.decode('ascii', errors='ignore')
                    precs.append(len(vs.split('.')[1]) if '.' in vs else 0)
            fp['coord_prec'] = round(sum(precs) / len(precs), 1)
        else:
            fp['coord_prec'] = -1
        t_ops = len(self._RE_T_OPS.findall(stream))
        g_ops = len(self._RE_G_OPS.findall(stream))
        fp['text_ratio'] = round(t_ops / max(t_ops + g_ops, 1), 2)
        return fp

    def _precision_shifts(self, streams: list[bytes | None]) -> list[SplitSignal]:
        signals = []
        precs = []
        for stream in streams:
            if not stream:
                precs.append(None)
                continue
            cs = self._RE_PREC_COORDS.findall(stream)
            if cs:
                ps = [len(c.decode('ascii', 'ignore').split('.')[1]) if b'.' in c else 0
                      for c in cs[:30]]
                precs.append(round(sum(ps) / len(ps), 1))
            else:
                precs.append(None)

        rl = 0
        for i in range(1, len(precs)):
            if precs[i] is None or precs[i - 1] is None:
                rl = 0; continue
            if abs(precs[i] - precs[i - 1]) < 0.5:
                rl += 1
            else:
                if rl >= 4:
                    signals.append(SplitSignal(
                        page_num=i, engine=self.NAME, confidence=0.4,
                        reason=f"Coordinate precision shift: {precs[i-1]} -> {precs[i]} decimals"
                    ))
                rl = 0
        return signals


# ─────────────────────────────────────────────
# Signal fusion and split decision
# ─────────────────────────────────────────────

class SplitDecider:
    """Fuses signals from all engines and determines final split points."""

    DEFAULT_WEIGHTS = {
        "bookmark": 2.5,
        "page_number": 2.0,
        "text_pattern": 1.8,
        "text_corpus": 1.6,
        "xref_structure": 1.0,
        "geometry": 1.2,
        "content_dialect": 1.0,
        "font_fingerprint": 0.8,
        "blank_separator": 0.7,
        "density": 0.6,
    }

    # Engines whose signal alone can justify a split (strong engines).
    # Weak/noisy engines should only contribute when a strong one also fires.
    STRONG_ENGINES = {"bookmark", "page_number", "text_pattern", "text_corpus"}

    # Minimum per-engine confidence to count toward the multi-engine bonus
    MIN_BONUS_CONFIDENCE = 0.45

    def __init__(self, total_pages: int, weights: dict = None,
                 threshold: float = 3.0, min_doc_pages: int = 4,
                 proximity_merge: int = 5, bookmark_labels: dict = None):
        self.total_pages = total_pages
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self.min_doc_pages = min_doc_pages
        self.proximity_merge = proximity_merge
        self.bookmark_labels = bookmark_labels or {}

    def decide(self, all_signals: list[SplitSignal],
               features: list[PageFeatures]) -> list[SplitPoint]:

        page_signals: dict[int, list[SplitSignal]] = defaultdict(list)
        for sig in all_signals:
            page_signals[sig.page_num].append(sig)

        candidates = []
        for page_num, sigs in page_signals.items():
            if page_num <= 0 or page_num >= self.total_pages:
                continue

            engine_scores = defaultdict(float)
            engine_reasons = {}
            for sig in sigs:
                if sig.confidence > engine_scores[sig.engine]:
                    engine_scores[sig.engine] = sig.confidence
                    engine_reasons[sig.engine] = sig.reason

            combined = sum(self.weights.get(e, 1.0) * c
                           for e, c in engine_scores.items())

            # Multi-engine bonus: only count engines with meaningful
            # confidence, and require at least one strong engine.
            qualifying = [e for e, c in engine_scores.items()
                          if c >= self.MIN_BONUS_CONFIDENCE]
            has_strong = any(e in self.STRONG_ENGINES for e in qualifying)
            n_qual = len(qualifying)

            if has_strong:
                if n_qual >= 4:
                    combined *= 1.4
                elif n_qual >= 3:
                    combined *= 1.25
                elif n_qual >= 2:
                    combined *= 1.1

            candidates.append(SplitPoint(
                page_num=page_num,
                combined_score=round(combined, 3),
                signals=[f"{e}: {engine_reasons[e]} (conf={engine_scores[e]:.2f})"
                         for e in sorted(engine_scores)],
                label=self._generate_label(page_num, features)
            ))

        candidates.sort(key=lambda sp: sp.combined_score, reverse=True)
        candidates = [c for c in candidates if c.combined_score >= self.threshold]
        candidates = self._merge_nearby(candidates)
        candidates = self._enforce_min_pages(candidates)
        candidates.sort(key=lambda sp: sp.page_num)
        return candidates

    def _merge_nearby(self, cands):
        if not cands:
            return cands
        merged = [cands[0]]
        for c in cands[1:]:
            if all(abs(c.page_num - k.page_num) > self.proximity_merge for k in merged):
                merged.append(c)
        return merged

    def _enforce_min_pages(self, cands):
        if not cands:
            return cands
        sc = sorted(cands, key=lambda c: c.page_num)
        valid = []
        prev = 0
        for c in sc:
            if c.page_num - prev >= self.min_doc_pages:
                valid.append(c)
                prev = c.page_num
        if valid and (self.total_pages - valid[-1].page_num) < self.min_doc_pages:
            valid.pop()
        return valid

    # Lines that are generic boilerplate and should be skipped when
    # generating a label from page text.
    _BOILERPLATE_RE = re.compile(
        r'^(?:'
        r'(?:confidential|proprietary|restricted|internal\s+use)'
        r'|(?:rev(?:ision)?\.?\s*[:\-]?\s*[a-z0-9]{1,4}$)'
        r'|(?:page\s*\d|sheet\s*\d)'
        r'|(?:date|prepared\s+by|checked\s+by|approved\s+by)'
        r'|(?:doc(?:ument)?\s*(?:no|#|number|ref))'
        r'|(?:project\s*(?:no|#|number)?)'
        r'|(?:(?:pmc|epcm|epc)\s*/?\s*(?:pmc|epcm|epc)?\s*services?\s)'
        r'|(?:client\s*[:\-])'
        r'|(?:owner\s*[:\-]?$)'
        r')',
        re.IGNORECASE,
    )

    # Patterns that look like useful document identifiers or titles
    _DOC_ID_RE = re.compile(
        r'(?:'
        r'(?:annex|appendix|attachment|schedule|volume|section|part)\s'
        r'|(?:specification|datasheet|data\s+sheet)'
        r'|(?:scope\s+of\s+(?:work|supply))'
        r'|(?:bill\s+of\s+(?:quantities|materials))'
        r'|(?:drawing|drg|dwg)\s'
        r'|(?:table\s+of\s+contents)'
        r'|(?:general\s+(?:terms|conditions|arrangement))'
        r'|\b\d{3,}[A-Z]?[\-/]\d{2,}'   # Doc number patterns like 077154C-000-...
        r')',
        re.IGNORECASE,
    )

    def _generate_label(self, page_num, features) -> str:
        feat = features[page_num] if page_num < len(features) else None

        # If the page has a section title (from header detection), prefer it
        # over a bookmark label since it's more descriptive
        if feat and feat.section_title:
            bm = self.bookmark_labels.get(page_num, "")
            if bm and bm.lower() not in feat.section_title.lower():
                return f"{feat.section_title[:50]} - {bm[:30]}"
            return feat.section_title[:80]

        if page_num in self.bookmark_labels:
            return self.bookmark_labels[page_num]

        if feat:
            lines = [l.strip() for l in feat.text.split('\n') if l.strip()]
            non_boilerplate = [l for l in lines[:12]
                               if len(l) > 3
                               and not re.match(r'^\d+$', l)
                               and not self._BOILERPLATE_RE.search(l)]
            # First prefer a line that looks like a document title/ID
            for line in non_boilerplate:
                if self._DOC_ID_RE.search(line):
                    return line[:80]
            # Then fall back to the first non-boilerplate line
            if non_boilerplate:
                return non_boilerplate[0][:80]
        return f"Document starting at page {page_num + 1}"


# ─────────────────────────────────────────────
# PDF Splitter orchestrator
# ─────────────────────────────────────────────

class PDFSplitter:
    """Main orchestrator: extracts features, runs engines, splits the PDF."""

    def __init__(self, pdf_path: str, output_dir: str = None,
                 threshold: float = 3.0, min_doc_pages: int = 4,
                 proximity_merge: int = 5, weights: dict = None,
                 dry_run: bool = False, report_only: bool = False):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir) if output_dir else self.pdf_path.parent / "split_output"
        self.threshold = threshold
        self.min_doc_pages = min_doc_pages
        self.proximity_merge = proximity_merge
        self.weights = weights
        self.dry_run = dry_run
        self.report_only = report_only

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def run(self) -> dict:
        log.info(f"Opening: {self.pdf_path}")
        doc = fitz.open(str(self.pdf_path))
        total_pages = len(doc)
        log.info(f"Total pages: {total_pages}")

        if total_pages < 4:
            log.info("PDF too short to split meaningfully.")
            doc.close()
            return {"status": "skipped", "reason": "too_few_pages",
                    "total_pages": total_pages}

        # Step 1: Extract features
        extractor = FeatureExtractor(doc)
        features = extractor.extract_all()

        # Step 2: Build bookmark label lookup for the decider
        bookmark_labels = self._build_bookmark_labels(doc)

        # Step 3: Run all detection engines
        log.info("Running detection engines...")
        engines = [
            BookmarkEngine(),
            GeometryEngine(),
            PageNumberEngine(),
            TextPatternEngine(),
            FontFingerprintEngine(),
            DensityEngine(),
            BlankPageEngine(),
            TextCorpusEngine(),
            XrefStructureEngine(),
            ContentStreamDialectEngine(),
        ]

        all_signals = []
        for engine in engines:
            all_signals.extend(engine.run(doc, features))

        log.info(f"Total signals collected: {len(all_signals)}")

        # Step 4: Fuse signals and decide split points
        decider = SplitDecider(
            total_pages=total_pages,
            weights=self.weights,
            threshold=self.threshold,
            min_doc_pages=self.min_doc_pages,
            proximity_merge=self.proximity_merge,
            bookmark_labels=bookmark_labels,
        )
        split_points = decider.decide(all_signals, features)

        log.info(f"Final split points: {len(split_points)}")
        for sp in split_points:
            log.info(f"  Page {sp.page_num + 1} (score={sp.combined_score:.2f}): {sp.label}")

        # Step 5: Generate report
        report = self._build_report(total_pages, features, all_signals, split_points)

        if self.report_only:
            doc.close()
            return report

        # Step 6: Split the PDF
        if not self.dry_run:
            self._execute_split(doc, split_points, total_pages, bookmark_labels)
        else:
            log.info("DRY RUN: No files written.")

        doc.close()
        return report

    def _build_bookmark_labels(self, doc) -> dict:
        """Extract bookmark titles keyed by 0-indexed page number.
        Prefers the deepest (most specific) bookmark title for labeling,
        since shallow entries are often generic project names."""
        labels = {}
        toc = doc.get_toc(simple=True)
        if not toc:
            return labels

        for level, title, page_num in toc:
            idx = page_num - 1
            if idx < 0:
                continue
            t = title.strip()
            if not t:
                continue
            # Keep the deepest (most specific) title per page.
            # When two bookmarks at the same level point to the same page,
            # prefer the longer (more descriptive) one.
            if idx not in labels:
                labels[idx] = (level, t)
            else:
                existing_level, existing_title = labels[idx]
                if level > existing_level:
                    labels[idx] = (level, t)
                elif level == existing_level and len(t) > len(existing_title):
                    labels[idx] = (level, t)

        # Flatten to just title strings
        return {idx: title for idx, (_, title) in labels.items()}

    def _build_report(self, total_pages, features, all_signals, split_points):
        segments = []
        boundaries = [0] + [sp.page_num for sp in split_points] + [total_pages]

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            label = ""
            if i > 0:
                label = split_points[i - 1].label
            else:
                seg_feat = features[start:end]
                if seg_feat:
                    lines = [l.strip() for l in seg_feat[0].text.split('\n') if l.strip()]
                    label = lines[0][:80] if lines else "First document"

            segments.append({
                "segment": i + 1,
                "start_page": start + 1,
                "end_page": end,
                "num_pages": end - start,
                "label": label,
                "score": split_points[i - 1].combined_score if i > 0 else None,
                "signals": split_points[i - 1].signals if i > 0 else [],
            })

        return {
            "status": "success",
            "input_file": str(self.pdf_path),
            "total_pages": total_pages,
            "split_points_found": len(split_points),
            "documents_produced": len(segments),
            "threshold_used": self.threshold,
            "engine_signal_counts": dict(Counter(s.engine for s in all_signals)),
            "segments": segments,
        }

    def _execute_split(self, doc, split_points, total_pages, bookmark_labels):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        boundaries = [0] + [sp.page_num for sp in split_points] + [total_pages]
        stem = self.pdf_path.stem
        toc = doc.get_toc(simple=True)

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]

            # Generate a clean filename from the label if available
            label_slug = ""
            if i > 0:
                raw_label = split_points[i - 1].label
            elif bookmark_labels.get(start):
                raw_label = bookmark_labels[start]
            else:
                raw_label = ""

            if raw_label:
                label_slug = re.sub(r'[^\w\s\-]', '', raw_label)[:50].strip()
                label_slug = re.sub(r'\s+', '_', label_slug)
                if label_slug:
                    label_slug = f"_{label_slug}"

            out_name = f"{stem}_part{i + 1:03d}_p{start + 1}-{end}{label_slug}.pdf"
            out_path = self.output_dir / out_name

            writer = fitz.open()
            writer.insert_pdf(doc, from_page=start, to_page=end - 1)

            # Preserve relevant bookmarks within this segment
            seg_toc = []
            for level, title, page_num in toc:
                if start + 1 <= page_num <= end:
                    new_page = page_num - start  # remap to new document
                    seg_toc.append([level, title, new_page])
            if seg_toc:
                # Normalize levels so the shallowest becomes level 1
                min_lvl = min(e[0] for e in seg_toc)
                for e in seg_toc:
                    e[0] = e[0] - min_lvl + 1
                try:
                    writer.set_toc(seg_toc)
                except Exception:
                    pass  # Some edge cases with single-page docs

            writer.save(str(out_path))
            writer.close()
            log.info(f"  Written: {out_name} ({end - start} pages, "
                     f"{len(seg_toc)} bookmarks)")

        log.info(f"All {len(boundaries) - 1} parts saved to: {self.output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Intelligently split concatenated EPC tender PDFs using heuristics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python splitter.py tender_package.pdf
  python splitter.py tender_package.pdf --threshold 0.6 --dry-run
  python splitter.py tender_package.pdf --report-only --report-json report.json
  python splitter.py tender_package.pdf -o /output/dir --min-pages 5
  python splitter.py tender_package.pdf --weights '{"bookmark": 3.0, "page_number": 2.5}'
        """)

    parser.add_argument("pdf", help="Path to the concatenated PDF file")
    parser.add_argument("-o", "--output-dir", help="Output directory for split PDFs")
    parser.add_argument("-t", "--threshold", type=float, default=3.0,
                        help="Min combined score for split point (default: 3.0)")
    parser.add_argument("--min-pages", type=int, default=4,
                        help="Min pages per split document (default: 4)")
    parser.add_argument("--proximity", type=int, default=5,
                        help="Merge splits within N pages (default: 5)")
    parser.add_argument("--weights", type=str, default=None,
                        help="JSON string of engine weight overrides")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze without writing files")
    parser.add_argument("--report-only", action="store_true",
                        help="Only produce the analysis report")
    parser.add_argument("--report-json", type=str, default=None,
                        help="Save report to a JSON file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError:
            log.error("Invalid JSON for --weights")
            sys.exit(1)

    splitter = PDFSplitter(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_doc_pages=args.min_pages,
        proximity_merge=args.proximity,
        weights=weights,
        dry_run=args.dry_run,
        report_only=args.report_only,
    )

    report = splitter.run()

    # Print summary
    print("\n" + "=" * 70)
    print("SPLIT ANALYSIS REPORT")
    print("=" * 70)
    print(f"Input:        {report.get('input_file', 'N/A')}")
    print(f"Total pages:  {report.get('total_pages', 'N/A')}")
    print(f"Split points: {report.get('split_points_found', 0)}")
    print(f"Documents:    {report.get('documents_produced', 0)}")
    print(f"Threshold:    {report.get('threshold_used', 'N/A')}")
    print()

    if "engine_signal_counts" in report:
        print("Engine signals:")
        for eng, count in sorted(report["engine_signal_counts"].items()):
            print(f"  {eng:25s}: {count}")
        print()

    if "segments" in report:
        print(f"{'Seg':>4} {'Pages':>12} {'Count':>6} {'Score':>7}  Label")
        print("-" * 70)
        for seg in report["segments"]:
            score_str = f"{seg['score']:.2f}" if seg['score'] is not None else "  ---"
            print(f"{seg['segment']:4d} {seg['start_page']:5d}-{seg['end_page']:<5d} "
                  f"{seg['num_pages']:6d} {score_str:>7s}  {seg['label'][:50]}")

    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report_json}")

    print()


if __name__ == "__main__":
    main()
