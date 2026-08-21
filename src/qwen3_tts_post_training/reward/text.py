"""Text normalization + CER, verbatim logic from playground/asr_rescore.py (bake-off validated).

Kills three known scoring bugs: NFKC variants, Chinese-numeral vs ITN arabic
("一九九八年" vs "1998年"), emoji/punctuation stripping ("好。😮").
Pure stdlib — usable in both trainer and scorer processes.
"""

import re
import unicodedata

CN_DIGIT = {
    "零": "0",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}
NUM_RE = re.compile(r"[零一二三四五六七八九十百千]+(?:点[零一二三四五六七八九十]+)?")
ARABIC_RE = re.compile(r"\d+(?:\.\d+)?")


def cn_value_parse(s: str) -> str | None:
    if not s:
        return None
    total, section, num = 0, 0, None
    for ch in s:
        if ch in CN_DIGIT:
            num = int(CN_DIGIT[ch]) if num is None else num * 10 + int(CN_DIGIT[ch])
        elif ch in CN_UNIT:
            u = CN_UNIT[ch]
            if num is None:
                num = 1
            if u == 10:
                section += num * 10
            else:
                section += num
                total += section * u if section else u
                section = 0
            num = None
        elif ch == "点":
            break
    if num is not None:
        section += num
    total += section
    return str(total)


def digit_by_digit(s: str) -> str:
    return "".join(CN_DIGIT.get(c, "") for c in s)


def canon_numeral(m: re.Match) -> str:
    s = m.group(0)
    if "点" in s:
        head, _, tail = s.partition("点")
        dec = "".join(CN_DIGIT.get(c, "") for c in tail)
        h = cn_value_parse(head) or digit_by_digit(head)
        return f"{h}.{dec}" if h else s
    return cn_value_parse(s) or digit_by_digit(s)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("百分之", "%")
    s = NUM_RE.sub(canon_numeral, s)
    out = []
    for c in s:
        if (c.isascii() and c.isalnum()) or "\u4e00" <= c <= "\u9fff":
            out.append(c)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ca = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = (
                prev[j - 1]
                if ca == b[j - 1]
                else 1 + min(prev[j - 1], prev[j], cur[j - 1])
            )
        prev = cur
    return prev[n]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
