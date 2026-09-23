"""Typeset the annotation guideline of record into the thesis appendix.

    python -m scripts.make_guideline_appendix

Source: `data/gold/calibration/_RULES.md` -- the rules the first (human-reviewed)
pass was held to, and, restated as a prompt, the rules the second annotator was
given. Output: `thesis/chapters/appendix_guideline.tex`, which
`thesis/chapters/appendix.tex` inputs.

Generated rather than transcribed on purpose. The guideline is amended as the
passes expose rules that read two ways (D-21 ... D-28 so far), and a hand-copied
appendix goes stale silently -- which is the same failure as the console and the
rules file drifting apart, one step further downstream.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "gold" / "calibration" / "_RULES.md"
OUT = ROOT / "thesis" / "chapters" / "appendix_guideline.tex"

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}
# characters that inputenc+T1 will not set on their own
_UNI = {"\u2026": r"\ldots{}", "\u2192": r"$\rightarrow$", "\u2264": r"$\leq$",
        "\u2265": r"$\geq$", "\ufffd": r"\texttt{?}", "\u00a0": "~"}


def esc(s: str) -> str:
    out = []
    for ch in s:
        if ch in _ESC:
            out.append(_ESC[ch])
        elif ch in _UNI:
            out.append(_UNI[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline(s: str) -> str:
    """Markdown inline -> LaTeX.

    Code spans are lifted out behind placeholders rather than split on, because
    emphasis in this file wraps around them (``**... `facts_hires`.**``) and a
    split leaves the two markers in different pieces, where no regex pairs them.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(r"\texttt{" + esc(m.group(1)) + "}")
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]*)`", stash, s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"\\emph{\1}", s, flags=re.S)
    return re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], s)


def unwrap(md: str) -> list[str]:
    """Join soft-wrapped lines back into one logical line per paragraph / item.

    Inline markup wraps across source lines in this file (`**on one condition:
    the\\nlabel must ...**`), and a line-at-a-time converter leaves those markers
    in the output. LaTeX does not need the source line breaks, so remove them
    before converting -- but never inside a fenced block, a table or a heading.
    """
    out: list[str] = []
    in_code = False
    for ln in md.splitlines():
        fence = ln.strip().startswith("```")
        special = (fence or in_code or not ln.strip()
                   or ln.lstrip().startswith("|")
                   or re.match(r"^#{1,3}\s", ln)
                   or re.match(r"^\d+\.\s", ln)
                   or re.match(r"^\s*[-*]\s", ln))
        if fence:
            in_code = not in_code
        if special or not out:
            out.append(ln.rstrip() if (in_code or fence) else ln.strip() or "")
            continue
        prev = out[-1]
        joinable = (prev.strip() and not prev.strip().startswith("```")
                    and not prev.lstrip().startswith("|")
                    and not re.match(r"^#{1,3}\s", prev))
        if joinable:
            out[-1] = prev.rstrip() + " " + ln.strip()
        else:
            out.append(ln.strip())
    return out


def convert(md: str) -> str:
    lines = unwrap(md)
    out: list[str] = []
    i, in_code, in_list = 0, False, False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r"\end{enumerate}")
            in_list = False

    while i < len(lines):
        ln = lines[i]

        if ln.strip().startswith("```"):
            if in_code:
                out.append(r"\end{Verbatim}")
            else:
                close_list()
                out.append(r"\begin{Verbatim}[fontsize=\small,xleftmargin=1em]")
            in_code = not in_code
            i += 1
            continue
        if in_code:
            out.append(ln)
            i += 1
            continue

        if not ln.strip():
            out.append("")
            i += 1
            continue

        if ln.lstrip().startswith("|"):
            close_list()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            out.append(r"\begin{center}\small")
            out.append(r"\begin{tabular}{@{}" + "l" * ncol + r"@{}}\toprule")
            for ri, r in enumerate(rows):
                r = r + [""] * (ncol - len(r))
                out.append(" & ".join(inline(c) for c in r) + r" \\"
                           + (r"\midrule" if ri == 0 else ""))
            out.append(r"\bottomrule\end{tabular}")
            out.append(r"\end{center}")
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            close_list()
            lvl = len(m.group(1))
            cmd = {1: r"\section*", 2: r"\subsection*", 3: r"\subsubsection*"}[lvl]
            out.append(f"{cmd}{{{inline(m.group(2))}}}")
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if m:
            if not in_list:
                out.append(r"\begin{enumerate}[leftmargin=2em,itemsep=0.4em]")
                in_list = True
            out.append(r"\item[\textbf{" + m.group(1) + r".}] " + inline(m.group(2)))
            i += 1
            continue

        if re.match(r"^\s*[-*]\s+", ln):
            body = re.sub(r"^\s*[-*]\s+", "", ln)
            out.append(r"\par\noindent$\bullet$~" + inline(body))
            i += 1
            continue

        # continuation of a list item, or a plain paragraph
        out.append(inline(ln.strip()))
        i += 1

    if in_code:
        out.append(r"\end{Verbatim}")
    close_list()
    return "\n".join(out)


def selfcheck(tex: str) -> list[str]:
    """Refuse to write LaTeX that will not compile.

    There is no pdflatex on the authoring machine, so a conversion bug would
    otherwise surface as a broken build weeks later, in a chapter nobody edits by
    hand. These are the failures this converter can actually produce: an
    unbalanced environment, an unbalanced brace, an unescaped special, and a
    leftover Markdown marker from an inline construct it did not recognise.
    """
    problems = []
    for env in ("enumerate", "Verbatim", "tabular", "center"):
        b = len(re.findall(r"\\begin\{" + env + r"\}", tex))
        e = len(re.findall(r"\\end\{" + env + r"\}", tex))
        if b != e:
            problems.append(f"{env}: {b} begin vs {e} end")

    depth = stray = i = 0
    while i < len(tex):
        c = tex[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                stray += 1
                depth = 0
        i += 1
    if depth or stray:
        problems.append(f"braces: depth {depth} at EOF, {stray} stray closer(s)")

    plain = re.sub(r"\\begin\{Verbatim\}.*?\\end\{Verbatim\}", "", tex, flags=re.S)
    for ch in "%#_":
        n = len(re.findall(r"(?<!\\)" + re.escape(ch), plain))
        if n:
            problems.append(f"{n} unescaped {ch!r}")
    if plain.count("$") % 2:
        problems.append("odd number of $ — unbalanced math mode")
    if "**" in plain or re.search(r"(?m)^\s*\|", plain) or re.search(r"(?m)^#{1,3} ", plain):
        problems.append("leftover Markdown marker (**, table row, or heading)")
    return problems


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    problems = selfcheck(body)
    if problems:
        for p in problems:
            print("  !!", p)
        raise SystemExit(f"{len(problems)} conversion problem(s) — nothing written")
    header = (
        "% GENERATED by scripts/make_guideline_appendix.py -- do not edit here.\n"
        "% Source: data/gold/calibration/_RULES.md\n"
        "% Regenerate after every amendment to the guideline.\n\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(header + body + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(body)} chars from {SRC.name})")
    print("  thesis/main.tex must load fancyvrb and enumitem (enumitem is already there)")


if __name__ == "__main__":
    main()
