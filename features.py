# -*- coding: utf-8 -*-
"""Handcrafted per-post features for FinArg-3 SM.

Lexicons target Taiwan stock-forum (PTT Stock / CMoney) Traditional Chinese.
Each feature is a cheap, interpretable signal hypothesized to correlate with
argument quality / professionalism (cf. Chen et al. WWW-2021, CIKM-2024).
"""
import math
import re

LEX = {
    # fundamental-analysis vocabulary (professional register)
    "fundamental": [
        "營收", "毛利", "毛利率", "淨利", "獲利", "財報", "法說", "股利",
        "配息", "配股", "本益比", "殖利率", "業績", "年增", "月增", "季增",
        "營益率", "產能", "訂單", "出貨", "市佔", "轉投資", "認列", "入帳",
        "基本面", "總銷", "完工", "交屋", "建案", "營運",
    ],
    "eps": ["EPS", "eps", "每股盈餘", "每股獲利"],
    # technical-analysis vocabulary
    "technical": [
        "均線", "年線", "季線", "月線", "K線", "k線", "型態", "支撐", "壓力",
        "突破", "跌破", "做頭", "打底", "背離", "量能", "爆量", "無量",
        "技術面", "缺口", "長紅", "長黑", "上影線", "下影線", "布林", "KD",
        "MACD", "RSI",
    ],
    # chips / institutional flows
    "chips": [
        "外資", "投信", "自營商", "主力", "籌碼", "融資", "融券", "借券",
        "大戶", "散戶", "買超", "賣超", "吃貨", "出貨",
    ],
    # bullish stance
    "bullish": [
        "看多", "做多", "買進", "加碼", "進場", "上看", "看好", "噴出",
        "漲停", "起漲", "續抱", "抱緊", "低接", "布局", "回升", "反彈",
        "創新高", "成長",
    ],
    # bearish stance
    "bearish": [
        "看空", "做空", "放空", "賣出", "減碼", "出場", "跌停", "崩",
        "破底", "下探", "跳水", "套牢", "停損", "轉弱", "利空", "重挫",
    ],
    # hedging / uncertainty (amateur register)
    "hedge": [
        "應該", "可能", "也許", "或許", "感覺", "好像", "大概", "猜",
        "夢到", "聽說", "不知道", "希望", "賭", "看看", "再看",
    ],
    # emotional / chatty markers
    "emotional": [
        "哈哈", "XD", "唉", "氣死", "傻眼", "嗚", "怒", "爽", "慘",
        "!!", "！！", "??", "？？", "~~", "～～",
    ],
    # explicit reasoning connectives (argument structure)
    "reasoning": [
        "因為", "所以", "因此", "由於", "導致", "預估", "推估", "預期",
        "假設", "若", "推測", "評估", "認為", "根據", "依據",
    ],
    # concrete time references
    "temporal": [
        "第一季", "第二季", "第三季", "第四季", "Q1", "Q2", "Q3", "Q4",
        "上半年", "下半年", "今年", "明年", "去年", "本月", "下月", "月底",
    ],
    # market-index / macro / futures talk — index posts have mechanically low
    # MPP (the index moves a few % while single stocks move tens of %)
    "index_macro": [
        "大盤", "台指", "加權指數", "期指", "期權", "未平倉", "融資水位",
        "道瓊", "費半", "美股", "那斯達克", "小那", "美指", "聯準會", "FED",
        "非農", "台股下周", "盤勢", "多方時間", "結算",
    ],
    # low-volatility instruments — upside capped regardless of argument
    "low_vol_instrument": [
        "特別股", "甲特", "定存股", "金控", "銀行股", "電信", "中華電",
        "殖利率", "存股",
    ],
    # turnaround / mean-reversion narrative (user hypothesis: high MPP)
    "turnaround": [
        "翻身", "轉型", "翻倍", "回升", "錯殺", "超跌", "利空出盡", "抄底",
        "落後補漲", "反轉", "止跌", "打底", "谷底",
    ],
}

RE_NUM = re.compile(r"\d+(?:\.\d+)?")
RE_PCT = re.compile(r"\d+(?:\.\d+)?%")
RE_MONEY = re.compile(r"\d+(?:\.\d+)?\s*(?:億|萬|元)")
RE_TICKER = re.compile(r"(?<!\d)[1-9]\d{3}(?!\d)")
RE_URL = re.compile(r"https?://\S+")


def _count(text: str, words) -> int:
    return sum(text.count(w) for w in words)


def post_features(text: str) -> dict:
    n = max(len(text), 1)
    feats = {
        "log_len": math.log(n),
        "num_count": len(RE_NUM.findall(text)),
        "num_density": len(RE_NUM.findall(text)) / n * 100,
        "pct_count": len(RE_PCT.findall(text)),
        "money_count": len(RE_MONEY.findall(text)),
        "ticker_count": len(
            [c for c in RE_TICKER.findall(text) if not 1990 <= int(c) <= 2030]
        ),
        "url_count": len(RE_URL.findall(text)),
        "question_marks": text.count("?") + text.count("？"),
        "exclam_marks": text.count("!") + text.count("！"),
    }
    for name, words in LEX.items():
        c = _count(text, words)
        feats[f"lex_{name}"] = c
        feats[f"lex_{name}_density"] = c / n * 100
    # stance balance: net bullishness
    feats["stance_net"] = feats["lex_bullish"] - feats["lex_bearish"]
    return feats


FEATURE_NAMES = sorted(post_features("測試 2330 目標價600元 EPS 5.5 因為營收年增20%").keys())


def feature_vector(text: str):
    f = post_features(text)
    return [f[k] for k in FEATURE_NAMES]
