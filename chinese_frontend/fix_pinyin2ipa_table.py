"""
Apply targeted corrections to a pinyin->IPA conversion table (pinyin2ipa.txt).

The shipped table has a handful of entries that inject noise into training:
a syllable with no vowel, an erhua row copied from the wrong initial, and an
erhua row using inconsistent symbols. This reads your table, rewrites just
those rows (leaving everything else byte-for-byte), and prints what changed.

Usage:
    python fix_pinyin2ipa_table.py path/to/pinyin2ipa.txt -o pinyin2ipa.fixed.txt
    # then diff/inspect, and replace the original when happy
"""
import argparse

# pinyin -> corrected IPA. Rationale in comments.
CORRECTIONS = {
    # "yun ɥˈn" had NO vowel. jun/qun/xun all use the -yn nucleus (tɕˈyn ...),
    # and yu is ˈy, so bare yun should be ˈyn.
    "yun":   "ˈyn",
    # "dour xˈɔɹ" mapped a d-initial erhua onto an x-(h-)initial. dou is tˈoʊ,
    # so its erhua is tˈoʊɹ.
    "dour":  "tˈoʊɹ",
    # "guanr gˈwɐʴ" used g/ʴ while every other g-initial uses k and every other
    # erhua uses ɹ. guan is kˈwan -> guanr is kˈwaɹ.
    "guanr": "kˈwaɹ",
    # "den tˈɚ" looked like an erhua; plain den (e.g. 扽) is tˈən.
    "den":   "tˈən",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table", help="path to pinyin2ipa.txt")
    ap.add_argument("-o", "--out", required=True, help="output path for the fixed table")
    args = ap.parse_args()

    seen = set()
    out_lines = []
    changed = []
    with open(args.table, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                parts = stripped.split()
                if len(parts) == 2 and parts[0] in CORRECTIONS:
                    py, old_ipa = parts
                    seen.add(py)
                    new_ipa = CORRECTIONS[py]
                    if new_ipa != old_ipa:
                        changed.append((py, old_ipa, new_ipa))
                        out_lines.append(f"{py} {new_ipa}\n")
                        continue
            out_lines.append(line if line.endswith("\n") else line + "\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"Wrote {args.out}")
    for py, old, new in changed:
        print(f"  {py}: {old}  ->  {new}")
    missing = set(CORRECTIONS) - seen
    if missing:
        print(f"  (not found in table, left alone: {sorted(missing)})")
    if not changed:
        print("  (no changes; table already matches the corrections)")


if __name__ == "__main__":
    main()
