"""
Drop-in replacement for hanzi_to_ipa() that groups syllables into prosodic words
so inference matches the BiaoBei-trained convention (space = word boundary,
phonemes contiguous within a word) instead of space-separating every syllable.

Difference from the original hanzi_to_ipa.py: it segments the text with jieba
first and phonemizes per word, emitting each word as one contiguous phoneme run
with a single space between words. Punctuation becomes a pause token. The
per-syllable IPA + tone mapping is unchanged, so this only alters spacing.

At training time you use the gold #1/#2 prosody boundaries (preprocess_biaobei.py);
at inference you don't have them, so jieba lexical words approximate the prosodic
words. It won't match BiaoBei exactly, but it gives the model real grouping
instead of the every-syllable-is-a-word layout.

Requires: pip install jieba  (plus your existing pypinyin + pinyin_ipa_converter)
"""
import os
import jieba
from pypinyin import lazy_pinyin, Style

from pinyin_ipa_converter import load_converter

_curdir = os.path.split(os.path.abspath(__file__))[0]


def map_punctuation(text: str):
    text = text.replace('、', ', ').replace('，', ', ')
    text = text.replace('。', '. ').replace('．', '. ')
    text = text.replace('！', '! ')
    text = text.replace('：', ': ')
    text = text.replace('；', '; ')
    text = text.replace('？', '? ')
    text = text.replace('«', ' “').replace('»', '” ')
    text = text.replace('《', ' “').replace('》', '” ')
    text = text.replace('「', ' “').replace('」', '” ')
    text = text.replace('【', ' “').replace('】', '” ')
    text = text.replace('（', ' “').replace('）', '” ')
    return text.strip()


def retone(p: str):
    p = p.replace('˧˩˧', '↓')   # third tone
    p = p.replace('˧˥', '↗')    # second tone
    p = p.replace('˥˩', '↘')    # fourth tone
    p = p.replace('˥', '→')     # first tone
    p = p.replace(chr(635) + chr(809), 'ɨ').replace(chr(633) + chr(809), 'ɨ')
    assert chr(809) not in p, p
    return p


# Point this at your pinyin2ipa.txt (same file the original module used).
converter = load_converter(os.path.join(_curdir, "pinyin2ipa.txt"))


def _is_hanzi_word(w: str) -> bool:
    return any('一' <= ch <= '鿿' for ch in w)


def hanzi_to_ipa(hanzi: str) -> str:
    """Convert Chinese text to grouped IPA: contiguous phonemes within each
    jieba word, single space between words, punctuation as pause tokens."""
    parts = []
    for w in jieba.cut(hanzi):
        w = w.strip()
        if not w:
            continue
        if _is_hanzi_word(w):
            pinyin = lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True,
                                 tone_sandhi=True, v_to_u=True)
            syls = []
            for py in pinyin:
                ipa = converter.convert(py)
                if ipa:
                    syls.append(retone(ipa))
                else:
                    # non-convertible piece inside a "word" (e.g. stray punct)
                    mapped = map_punctuation(py).strip()
                    if mapped:
                        syls.append(mapped)
            if syls:
                parts.append(''.join(syls))   # contiguous within the word
        else:
            mapped = map_punctuation(w).strip()
            if mapped:
                parts.append(mapped)           # punctuation -> its own pause token
    return ' '.join(parts)


if __name__ == "__main__":
    for s in ["卡尔普陪外孙玩滑梯。", "宝马配挂跛骡鞍，貂蝉怨枕董翁榻。"]:
        print(s, "->", hanzi_to_ipa(s))
