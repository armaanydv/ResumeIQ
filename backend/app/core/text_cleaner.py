"""
ResumeIQ Text Sanitization and Section Segmentation Utilities.
Normalizes messy PDF/Word characters and segments resumes into structured sections.
"""

import re
import unicodedata
from typing import Dict, List, Tuple


# Regex patterns for matching common resume section headings
SECTION_PATTERNS: Dict[str, List[str]] = {
    "summary": [
        r"^(?:professional\s+)?summary",
        r"^career\s+objective",
        r"^objective",
        r"^profile",
        r"^about\s+me",
    ],
    "skills": [
        r"^(?:technical\s+|core\s+)?skills(?:\s*&\s*technologies)?",
        r"^skills\s*&\s*competencies",
        r"^technical\s+proficiencies",
        r"^technologies(?:\s+used)?",
        r"^tech\s+stack",
        r"^tools\s*&\s*technologies",
    ],
    "experience": [
        r"^(?:work|professional|employment)\s+experience",
        r"^experience",
        r"^work\s+history",
        r"^employment\s+history",
        r"^career\s+history",
    ],
    "projects": [
        r"^(?:key\s+|featured\s+|personal\s+|academic\s+)?projects",
        r"^portfolio(?:\s+projects)?",
    ],
    "education": [
        r"^education(?:\s+and\s+qualifications)?",
        r"^academic\s+background",
        r"^academic\s+credentials",
        r"^qualifications",
    ],
    "certifications": [
        r"^certifications?(?:\s*&\s*licenses)?",
        r"^licenses?(?:\s*&\s*certifications)?",
        r"^courses?(?:\s*&\s*training)?",
        r"^professional\s+development",
    ],
}


def clean_text(raw_text: str) -> str:
    """
    Sanitize and normalize text extracted from PDF or DOCX documents.
    - Strips non-printable and zero-width characters.
    - Normalizes Unicode ligatures, smart quotes, dashes, and bullet points.
    - Preserves structural paragraphs and line boundaries.
    """
    if not raw_text:
        return ""

    # Normalize unicode to decomposed / composed canonical format
    text = unicodedata.normalize("NFKC", raw_text)

    # Remove zero-width characters and control characters except \n and \t
    text = re.sub(r"[\u200b-\u200f\ufeff\u00ad]", "", text)

    # Normalize typographic smart quotes to standard quotes
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)

    # Normalize typographic dashes (en-dash, em-dash, horizontal bar) to hyphen
    text = re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)

    # Normalize non-breaking spaces and tabs to regular spaces
    text = text.replace("\u00a0", " ").replace("\t", "    ")

    # Normalize miscellaneous bullet symbols to standard markdown hyphen bullet
    bullet_chars = r"[\u2022\u2023\u25e6\u2043\u2219\u25cf\u25aa\u25cb\u25b8\u25ba\u27a2\u2714\u2713]"
    text = re.sub(bullet_chars + r"\s*", "- ", text)

    # Split lines and clean trailing whitespaces
    cleaned_lines = []
    for line in text.splitlines():
        line_stripped = line.strip()
        # If line was a bullet with trailing space or punctuation
        if line_stripped:
            # Clean consecutive spaces within a line
            line_normalized = re.sub(r" {2,}", " ", line_stripped)
            cleaned_lines.append(line_normalized)
        else:
            # Keep empty line marker
            cleaned_lines.append("")

    # Rejoin lines
    result = "\n".join(cleaned_lines)

    # Reduce runs of 3+ newlines down to 2 (paragraph boundary)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def segment_sections(text: str) -> Dict[str, str]:
    """
    Segment clean resume text into standardized sections:
    - general (contact info/header before first heading)
    - summary
    - skills
    - experience
    - projects
    - education
    - certifications

    Returns a dictionary mapping section keys to their corresponding text content.
    """
    if not text:
        return {}

    lines = text.splitlines()
    detected_headings: List[Tuple[int, str]] = []  # (line_index, section_key)

    for idx, line in enumerate(lines):
        line_clean = line.strip().lower()

        # Section headings are usually short (under 50 chars) and standalone
        if not line_clean or len(line_clean) > 50:
            continue

        # Strip trailing punctuation like colons, hyphens, underlines
        line_label = re.sub(r"[:\-_#=*]+$", "", line_clean).strip()

        # Check against each section category
        matched_section = None
        for section_key, patterns in SECTION_PATTERNS.items():
            for pat in patterns:
                if re.match(pat, line_label):
                    matched_section = section_key
                    break
            if matched_section:
                break

        if matched_section:
            detected_headings.append((idx, matched_section))

    # If no headings detected, classify entire text as general
    if not detected_headings:
        return {"general": text.strip()}

    sections: Dict[str, List[str]] = {}

    # Content before the first detected section heading is 'general' (header/contact)
    first_idx, first_section = detected_headings[0]
    if first_idx > 0:
        header_text = "\n".join(lines[:first_idx]).strip()
        if header_text:
            sections["general"] = [header_text]

    # Slice content between heading indices
    for i in range(len(detected_headings)):
        curr_idx, curr_section = detected_headings[i]
        next_idx = detected_headings[i + 1][0] if (i + 1 < len(detected_headings)) else len(lines)

        # Content of current section starts on the line after the heading
        content_lines = lines[curr_idx + 1 : next_idx]
        section_content = "\n".join(content_lines).strip()

        if curr_section not in sections:
            sections[curr_section] = []
        if section_content:
            sections[curr_section].append(section_content)

    # Flatten lists of text per section
    return {k: "\n\n".join(v).strip() for k, v in sections.items() if v}
