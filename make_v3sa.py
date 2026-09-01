# -*- coding: utf-8 -*-
"""Generate llm_judge_v3sa.py: rubric v3 with ONLY definitional stance fixes.

The v3sa variant used in the paper's stance-correction analysis differs from
rubric v3 exclusively in its definitional statements: the payoff definition,
the bearish rule, the momentum rule, the output-format wording, and the
rationales of two few-shot exemplars. All few-shot pairs and labels, every
other rule, and all runtime parameters are identical.

Every replacement asserts exactly one occurrence, so any drift in the source
file fails loudly instead of silently producing a wrong variant.

Usage:  python make_v3sa.py     (writes llm_judge_v3sa.py next to llm_judge.py)
"""
from pathlib import Path

SRC = Path(__file__).parent / "llm_judge.py"
DST = Path(__file__).parent / "llm_judge_v3sa.py"

text = SRC.read_text(encoding="utf-8")

REPL = [
    # docstring header
    (
        '"""Track 1: LLM-as-judge for pairwise MPP comparison.',
        '"""Track 1 variant v3sa: rubric v3 with ONLY the stance handling corrected\n'
        "to the official stance-aware MPP definition (bull: max-high upside; bear:\n"
        "min-low downside). All other rules, few-shot pairs and labels, and params\n"
        "are identical to llm_judge.py (v3). Post-hoc diagnostic; frozen system untouched.",
    ),
    # task statement + key rule
    (
        "你會看到兩篇台灣股市社群貼文（貼文A與貼文B）。你的任務：判斷「在發文當下買進\\\n"
        "哪一篇貼文討論的股票」，之後一段時間內的最大可能獲利（MPP）較高。\n"
        "\n"
        "【關鍵規則】MPP的定義是「發文時買進、在未來窗口內的最高價賣出」的報酬率——\\\n"
        "一律做多、與貼文立場無關。所以你要預測的是「哪檔股票接下來比較會漲」，\\\n"
        "而不是「哪篇文章寫得比較好」。",
        "你會看到兩篇台灣股市社群貼文（貼文A與貼文B）。你的任務：判斷「在發文當下順著\\\n"
        "貼文立場操作哪一篇貼文討論的股票」，之後一段時間內的最大可能獲利（MPP）較高。\n"
        "\n"
        "【關鍵規則】MPP依貼文立場計算：看多文＝「發文時買進、在窗口內最高價賣出」的\\\n"
        "報酬率；看空文＝「發文時放空、在窗口內最低價回補」的報酬率。所以你要預測的是\\\n"
        "「哪篇貼文的方向判斷之後會兌現得更多」——看多文比誰漲得多、看空文比誰跌得深——\\\n"
        "而不是「哪篇文章寫得比較好」。",
    ),
    # rule 1: momentum scored by stance
    (
        "1. 貼文會洩漏當下的價格動能與市場情緒，優先讀取這些線索：抱怨「賣掉就漲、氣死」\\\n"
        "＝股票正在漲（高MPP訊號）；哀嚎「跌得真慘、套牢」＝正在跌（低MPP訊號）。",
        "1. 貼文會洩漏當下的價格動能與市場情緒，優先讀取這些線索：抱怨「賣掉就漲、氣死」\\\n"
        "＝股票正在漲；哀嚎「跌得真慘、套牢」＝正在跌。動能依立場計分：看多文遇上漲動能\\\n"
        "＝高MPP訊號，看空文遇下跌動能＝高MPP訊號；動能方向與立場相反則MPP偏低。",
    ),
    # rule 4: bearish scored on downside excursion
    (
        "4. 【看空文的相對評估】MPP一律做多計算：看空且理由紮實→股價傾向跌→MPP低。\\\n"
        "但比較是相對的——對手若是大盤文或訊號更弱的標的，看空文仍可能勝出。\\\n"
        "散戶的情緒性看空、停損投降文常出現在底部，反而可能是反轉訊號。",
        "4. 【看空文的評估】看空文以下檔空間計分：標的之後跌得越深，MPP越高。\\\n"
        "判斷其看空論點會不會兌現——看空且理由紮實、標的確實轉弱→下檔兌現→MPP高；\\\n"
        "逆勢喊空、下檔難兌現→MPP低。散戶的情緒性看空、停損投降文常出現在底部，\\\n"
        "此時剩餘下檔空間可能已經有限。",
    ),
    # output format wording
    (
        '請輸出JSON格式：{"analysis": "比較兩檔標的的上漲潛力（50字內）"',
        '請輸出JSON格式：{"analysis": "依各自立場比較兩篇的MPP潛力（50字內）"',
    ),
    # few-shot #2 rationale (label unchanged)
    (
        '"analysis": "A實質看空且股價正下跌，做多MPP必低；B雖粗糙但有除權息催化劑、散戶恐慌沈澱籌碼的反轉邏輯。",',
        '"analysis": "A看空但股價已從120跌到100、跌深後剩餘下檔空間有限；B看多且有除權息催化劑、散戶恐慌沈澱籌碼的反轉邏輯，上檔較可期。",',
    ),
    # few-shot #3 rationale (label unchanged)
    (
        '"analysis": "A看多且有獲利上修、主力吃貨、殖利率保護，動能與催化劑俱足；B全面看空，做多MPP低。",',
        '"analysis": "A看多且有獲利上修、主力吃貨、殖利率保護，動能與催化劑俱足；B看空但僅口號式總經擔憂、無標的轉弱的具體證據，下檔難兌現。",',
    ),
]

for old, new in REPL:
    n = text.count(old)
    assert n == 1, f"expected 1 occurrence, found {n}:\n{old[:80]}..."
    text = text.replace(old, new)

DST.write_text(text, encoding="utf-8")
print(f"wrote {DST.name} with {len(REPL)} replacements")
