"""VOXCPM KHMER TTS Text Preparation Engine."""

from __future__ import annotations
import re
import unicodedata

# Khmer punctuation -> Western equivalents (Section 9)
_KHM_PUNCT_MAP = {
    chr(0x17D4): ".",   # ។  Khmer full stop
    chr(0x17D6): ",",   # ៖  Khmer colon -> comma
    chr(0x17D7): "",    # ៗ  iteration mark -> removed
    chr(0x17D8): "",    # ៘  abbreviation -> removed
    chr(0x17D5): ".",   # ៕  closing circle
}

# Zero-width / invisible characters to strip
_ZW_CHARS = (chr(0x200B) + chr(0x200C) + chr(0x200D)
             + chr(0x200E) + chr(0x200F) + chr(0xFEFF) + chr(0x00AD))

# Khmer numerals -> Western digits
_KH_DIGITS = {chr(0x17E0 + i): str(i) for i in range(10)}

# Double-quote for dialogues
_DQ = chr(0x201D)

MAX_SENTENCE = 80
MAX_PHRASE = 45

# Khmer conjunctions (Section 4, 12)
_C = chr  # shorthand
_KHMER_CONJUNCTIONS = sorted([
    _C(0x17A0)+_C(0x17BE)+_C(0x1799),          # ហើយ
    _C(0x1794)+_C(0x17C9)+_C(0x17BB)+_C(0x1793)+_C(0x17D2)+_C(0x178F)+_C(0x17C2),  # ប៉ុន្តែ
    _C(0x1791)+_C(0x17C4)+_C(0x17C7)+_C(0x17A2)+_C(0x179F)+_C(0x17CB),  # ទោះអស់
    _C(0x1795)+_C(0x17D2)+_C(0x179F)+_C(0x17C1)+_C(0x1784)+_C(0x1791)+_C(0x17C0)+_C(0x178F),  # ផ្សេងទៀត
    _C(0x178A)+_C(0x17BC)+_C(0x1785)+_C(0x17D2)+_C(0x1793)+_C(0x17C1)+_C(0x17C7),  # ដូច្នេះ
    _C(0x178A)+_C(0x17C4)+_C(0x1799)+_C(0x179F)+_C(0x17B6)+_C(0x179A),  # ដោយសារ
    _C(0x1796)+_C(0x17D2)+_C(0x179A)+_C(0x17C4)+_C(0x17C7),  # ព្រោះ
    _C(0x1796)+_C(0x17D2)+_C(0x179A)+_C(0x17C4)+_C(0x17CF),  # ព្រោ៏
    _C(0x178A)+_C(0x17C4)+_C(0x1799),  # ដោយ
    _C(0x1781)+_C(0x178E)+_C(0x17C8),  # ខណៈ
    _C(0x1796)+_C(0x17C1)+_C(0x179B),  # ពេល
    _C(0x178A)+_C(0x17C2)+_C(0x179B),  # ដែល
    _C(0x17AB),  # ឫ
    _C(0x17A0)+_C(0x17B6)+_C(0x1785),  # ហាច
    _C(0x17A0)+_C(0x17BE)+_C(0x1799)+_C(0x1791)+_C(0x17C0)+_C(0x178F),  # ហើយទៀត
], key=len, reverse=True)


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------
def unicode_normalize(text: str) -> str:
    """NFC normalize, strip zero-width/invisible chars, fix whitespace."""
    text = unicodedata.normalize("NFC", text)
    for ch in _ZW_CHARS:
        text = text.replace(ch, "")
    # Normalize fullwidth ASCII to halfwidth
    text = "".join(
        chr(ord(c) - 0xFEE0) if 0xFF01 <= ord(c) <= 0xFF5E else c
        for c in text
    )
    # Standardize whitespace variants to a regular space
    _ws = chr(0x9) + chr(0xA0) + chr(0x2000) + "-" + chr(0x200B) + chr(0x202F) + chr(0x205F) + chr(0x3000)
    text = re.sub(r"[" + _ws + r"]+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Punctuation normalization (Section 9 + Section 3)
# ---------------------------------------------------------------------------
def normalize_punctuation(text: str) -> str:
    """Map Khmer punctuation to Western equivalents."""
    for kh, west in _KHM_PUNCT_MAP.items():
        text = text.replace(kh, west)
    return text


# --- Khmer word constants for regex patterns (using chr to avoid encoding) ---
_KH_SA = _C(0x179F)+_C(0x17BD)+_C(0x179A)+_C(0x1790)+_C(0x17B6)       # សួរថា
_KH_TA = _C(0x1790)+_C(0x17B6)                                          # ថា
_KH_NP = _C(0x1793)+_C(0x1797)+_C(0x17C2)+_C(0x1780)                   # នភែក
_KH_BNM = (_C(0x1794)+_C(0x1793)+_C(0x17D2)+_C(0x1791)+_C(0x17B6)+_C(0x1799)
           +_C(0x1798)+_C(0x1780)+_C(0x1790)+_C(0x17B6))               # បន្ទាយមកថា
_KH_KH_WORD = _C(0x1781)+_C(0x17C2)                                     # ខែ
_KH_CURRENCY = _C(0x17DB)                                               # ៛


# ---------------------------------------------------------------------------
# Dialogue handling (Section 5)
# ---------------------------------------------------------------------------
def handle_dialogue(text: str) -> str:
    """Detect and clearly separate dialogue from narration."""
    _quote_map = [
        (_C(0x300C), _DQ), (_C(0x300D), _DQ),
        (_C(0xFF08), "("), (_C(0xFF09), ")"),
        (_C(0x201C), _DQ), (_C(0x201D), _DQ),
        (_C(0x2018), _C(0x2019)),
    ]
    for asq, west in _quote_map:
        text = text.replace(asq, west)

    paragraphs = text.split("\n")
    processed: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            processed.append("")
            continue
        if para.startswith(_DQ):
            if para[-1] not in ".?!":
                para += "."
            processed.append(para)
            continue
        sentences = re.split(r"(?<=[.])\s+", para)
        new_sentences: list[str] = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Pattern: narration + [សួរថា | ថា | នភែក | បន្ទាយមកថា] + <speech>
            dialog_pat = (r'^(.*?)(?:\s+' + _KH_SA + r'\s+|\s+' + _KH_TA
                          + r'\s+|' + _KH_NP + r'\s+|' + _KH_BNM + r'\s+)'
                          + r'(.+?)(?:\s*[.])$')
            m = re.search(dialog_pat, sent)
            if m:
                narration = m.group(1).strip()
                dialogue = m.group(2).strip()
                if narration.endswith("."):
                    narration = narration[:-1].strip()
                if dialogue and dialogue[-1] not in ".?!":
                    dialogue += "."
                new_sentences.append(narration + "  " + _DQ + dialogue + _DQ)
                continue
            new_sentences.append(sent)
        processed.append(" ".join(new_sentences) if new_sentences else para)

    return "\n".join(processed)


# ---------------------------------------------------------------------------
# Sentence optimization (Sections 3, 4, 12)
# ---------------------------------------------------------------------------
_KH_PR = _C(0x1796)+_C(0x17D2)+_C(0x179A)+_C(0x17C4)+_C(0x17C7)
_KH_PR2 = _C(0x1796)+_C(0x17D2)+_C(0x179A)+_C(0x17C4)+_C(0x17CF)
_KH_DOI = _C(0x178A)+_C(0x17C4)+_C(0x1799)
_KH_DOISAR = _C(0x178A)+_C(0x17C4)+_C(0x1799)+_C(0x179F)+_C(0x17B6)+_C(0x179A)
_KH_TOOH = _C(0x1791)+_C(0x17C4)+_C(0x17C7)+_C(0x17A2)+_C(0x179F)+_C(0x17CB)


def split_conjunctions(sentence: str) -> list[str]:
    """Split a long sentence at natural Khmer conjunction pause points."""
    if len(sentence) <= 40:
        return [sentence]
    conj = "|".join(re.escape(c) for c in _KHMER_CONJUNCTIONS)
    pattern = re.compile(r"(?<=[.])\s+(" + conj + r")\s+")
    parts: list[str] = []
    last_end = 0
    current = ""
    for m in pattern.finditer(sentence):
        before = sentence[last_end:m.start()]
        current += before + " " + m.group(1) + " "
        if len(current) > MAX_PHRASE:
            parts.append(current.strip())
            current = ""
        last_end = m.end()
    remainder = sentence[last_end:].strip()
    if remainder:
        current += remainder
    if current.strip():
        parts.append(current.strip())
    if len(parts) == 1 and len(parts[0]) > MAX_SENTENCE:
        parts = split_at_clauses(parts[0])
    return parts if parts else [sentence]


def split_at_clauses(sentence: str) -> list[str]:
    """Fallback: split a long sentence at clause boundaries."""
    clause_words = (_KH_PR + "|" + _KH_PR2 + "|" + _KH_DOI + "|"
                    + _KH_DOISAR + "|" + _KH_TOOH)
    clauses = re.split(r"\s+(" + clause_words + r")\s+", sentence)
    result: list[str] = []
    for i, chunk in enumerate(clauses):
        if not chunk.strip():
            continue
        if i % 2 == 1:
            if result:
                result[-1] = result[-1] + " " + chunk + " "
            else:
                result.append(chunk)
        else:
            if result:
                result[-1] = result[-1] + " " + chunk
            else:
                result.append(chunk)
    return result if len(result) > 1 else [sentence]


def optimize_sentences(text: str) -> str:
    """Break long sentences into natural speech units."""
    paragraphs = text.split("\n")
    result_paras: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            result_paras.append("")
            continue
        sentences = re.split(r"(?<=[.])\s+", para)
        optimized: list[str] = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= MAX_PHRASE:
                optimized.append(sent)
                continue
            fragments = split_conjunctions(sent)
            optimized.extend(fragments)
        result_paras.append(" ".join(optimized))

    return "\n\n".join(result_paras)


# ---------------------------------------------------------------------------
# Number normalization (Section 7)
# ---------------------------------------------------------------------------
def normalize_numbers(text: str) -> str:
    """Convert Khmer numerals to Western digits and normalize formats."""
    for kh, west in _KH_DIGITS.items():
        text = text.replace(kh, west)
    # Normalize spaced Khmer dates: DD ខែMonth YYYY
    date_pat = r"(\d{1,4})\s+(" + _KH_KH_WORD + r"\S+)\s+(\d{1,4})"
    text = re.sub(date_pat, r"\1 \2 \3", text)
    # Phone numbers: group digits for easier speech
    text = re.sub(r"\b(\d{3})(\d{4,})\b", r"\1-\2", text)
    # Currency: ensure space before ៛ symbol
    text = re.sub(r"(\d)\s*" + _KH_CURRENCY, r"\1 " + _KH_CURRENCY, text)
    return text


# ---------------------------------------------------------------------------
# Paragraph organization (Section 10)
# ---------------------------------------------------------------------------
def organize_paragraphs(text: str) -> str:
    """Organize text into logical paragraphs with natural transitions."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n")]
    paragraphs = [p for p in paragraphs if p]
    # Merge fragments shorter than 20 chars into previous paragraph
    merged: list[str] = []
    for p in paragraphs:
        if p and len(p) < 20 and merged:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # Ensure dialogue paragraphs stand alone
    final: list[str] = []
    for p in merged:
        if p.startswith(_DQ):
            if final and final[-1] and not final[-1].startswith(_DQ):
                final.append("")
        final.append(p)
    return "\n".join(final)


# ---------------------------------------------------------------------------
# Final punctuation cleanup (Section 9)
# ---------------------------------------------------------------------------
def cleanup_punctuation(text: str) -> str:
    """Final cleanup of punctuation and whitespace artifacts."""
    text = re.sub(r"\s+([.!?,])", r"\1", text)
    khmer_char = r"[\u1780-\u17ff]"
    text = re.sub(r"([.!?,])\s*([chr(0x17E1)-chr(0x17E9)0-9a-zA-Z" + khmer_char + r"])", r"\1 \2", text)
    text = re.sub(r" {2,}", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------
class KhmerTextPreparationEngine:
    """Prepare raw Khmer text for VoxCPM2 TTS (all 14 rules).

    Pipeline:
    1. Unicode normalization (NFC) + invisible char removal
    2. Punctuation translation (Khmer -> Western)
    3. Dialogue detection & formatting
    4. Sentence splitting & phrase optimization
    5. Number / symbol normalization
    6. Paragraph reorganization
    7. Final punctuation cleanup
    """

    def __init__(self) -> None:
        pass

    def prepare(self, text: str) -> str:
        """Run the full preparation pipeline."""
        if not text or not text.strip():
            return ""
        text = unicode_normalize(text)
        text = normalize_punctuation(text)
        text = handle_dialogue(text)
        text = optimize_sentences(text)
        text = normalize_numbers(text)
        text = organize_paragraphs(text)
        text = cleanup_punctuation(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_engine = KhmerTextPreparationEngine()


def prepare_khmer_text(text: str) -> str:
    """Prepare raw Khmer text for VoxCPM / VoxCPM2 TTS."""
    return _engine.prepare(text)


