"""Conservative, dependency-free CSS / JS minifiers used by ``scripts/build_frontend.py``.

Deliberately NOT a full minifier (no renaming, no rewriting): it only removes what can never change
behaviour -- comments and insignificant whitespace -- and it preserves string / template-literal /
regex-literal contents byte for byte. JS keeps its newlines (so automatic semicolon insertion keeps
working); only comments, blank lines and leading indentation outside template literals are dropped.

The output is validated in tests (``node --check`` on every JS file; CSS brace balance).
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- CSS

def minify_css(src: str) -> str:
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            # a comment separates tokens (e.g. `a/**/b`): keep one space if needed
            if out and out[-1] not in " {};:,>" and i < n and src[i] not in " \n\t{};:,>":
                out.append(" ")
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if c in " \t\r\n":
            j = i
            while j < n and src[j] in " \t\r\n":
                j += 1
            prev = out[-1][-1] if out else ""
            nxt = src[j] if j < n else ""
            if prev and prev not in "{};:,>" and nxt and nxt not in "{};,>":
                out.append(" ")
            i = j
            continue
        if c == ";" and re.match(r"\s*\}", src[i + 1:i + 9]):
            i += 1          # drop the last `;` before `}`
            continue
        out.append(c)
        i += 1
    css = "".join(out)
    # `calc(1px + 2px)` and friends need their spaces; we only ever removed whitespace next to
    # { } ; : , > which are safe. A space after `:` is dropped only in declarations -- but selectors
    # like `a :hover` / `a: hover` do not occur with a space before the colon here (checked in tests).
    return css.strip()


# --------------------------------------------------------------------------- JS

_REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {None}
_REGEX_KW = {"return", "typeof", "case", "in", "of", "delete", "void", "throw", "new", "else", "do", "yield", "await"}


def minify_js(src: str) -> str:
    out: list[str] = []
    i, n = 0, len(src)
    # stack of template-literal brace depths: entering `${` pushes the depth, matching `}` resumes the template
    tpl_stack: list[int] = []
    depth = 0
    last_sig: str | None = None       # last significant char (for regex detection)
    last_word = ""
    line_start = True                  # at beginning of a line (drop indentation)

    def emit(s: str) -> None:
        out.append(s)

    def read_template(j: int) -> int:
        """Copy template text from j (just after the opening backtick or a closing `}` of ${}) verbatim."""
        nonlocal depth
        k = j
        while k < n:
            ch = src[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "`":
                emit(src[j:k + 1])
                return k + 1
            if ch == "$" and src[k + 1:k + 2] == "{":
                emit(src[j:k + 2])
                tpl_stack.append(depth)
                depth += 1
                return k + 2
            k += 1
        emit(src[j:])
        return n

    while i < n:
        c = src[i]
        if c == "\n":
            if out and out[-1] != "\n":
                emit("\n")
            i += 1
            line_start = True
            continue
        if c in " \t\r":
            j = i
            while j < n and src[j] in " \t\r":
                j += 1
            if not line_start and j < n and src[j] != "\n":
                emit(" ")
            i = j
            continue
        line_start = False
        if c == "/" and src[i + 1:i + 2] == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and src[i + 1:i + 2] == "*":
            j = src.find("*/", i + 2)
            seg = src[i:(n if j < 0 else j + 2)]
            i = n if j < 0 else j + 2
            if "\n" in seg:                 # keep a newline so ASI semantics are unchanged
                if out and out[-1] != "\n":
                    emit("\n")
                line_start = True
            else:
                emit(" ")
            continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            emit(src[i:j + 1])
            i = j + 1
            last_sig, last_word = c, ""
            continue
        if c == "`":
            emit("`")
            i = read_template(i + 1)
            last_sig, last_word = "`", ""
            continue
        if c == "}" and tpl_stack and depth - 1 == tpl_stack[-1]:
            tpl_stack.pop()
            depth -= 1
            emit("}")
            i = read_template(i + 1)
            last_sig, last_word = "`", ""
            continue
        if c == "/":
            is_regex = last_sig in _REGEX_PREV or (last_word in _REGEX_KW and last_sig is not None and last_sig.isalnum())
            if is_regex:
                j, in_class = i + 1, False
                while j < n:
                    ch = src[j]
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == "[":
                        in_class = True
                    elif ch == "]":
                        in_class = False
                    elif ch == "/" and not in_class:
                        break
                    elif ch == "\n":
                        break
                    j += 1
                j += 1
                while j < n and (src[j].isalpha()):
                    j += 1
                emit(src[i:j])
                i = j
                last_sig, last_word = ")", ""     # a regex literal acts like a value
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            last_word = src[i:j]
            last_sig = src[j - 1]
            emit(last_word)
            i = j
            continue
        emit(c)
        last_sig, last_word = c, ""
        i += 1
    res = "".join(out)
    res = re.sub(r"\n{2,}", "\n", res)
    return res.strip() + "\n"
