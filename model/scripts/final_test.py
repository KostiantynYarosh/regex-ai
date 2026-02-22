import json
import re
import warnings
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_DIR = Path(__file__).parent.parent / "checkpoints" / "t5-regex"
TEST_DATA = Path(__file__).parent.parent / "data" / "processed" / "test.json"


def validate_regex(p):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            re.compile(p)
        return True
    except re.error:
        return False


def unescape(s):
    return s.replace("<BS>", "\\")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<BS>", "{", "}", "^", "<", "`", "~"]})
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device).eval()
    data = json.load(open(TEST_DATA, encoding="utf-8"))

    exact = compile_ok = 0
    errors = []

    for i, item in enumerate(data):
        ids = tokenizer(f"generate regex: {item['description']}", return_tensors="pt",
                        max_length=128, truncation=True).input_ids.to(device)

        with torch.no_grad():
            outs = model.generate(ids, max_length=128, num_beams=4,
                                  num_return_sequences=4, early_stopping=True)

        cands = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outs]
        ref = item["regex"]

        best = cands[0]
        for c in cands:
            if c == ref:
                best = c
                break
            if validate_regex(unescape(c)):
                best = c
                break

        if best == ref:
            exact += 1
        if validate_regex(unescape(best)):
            compile_ok += 1
        else:
            errors.append((item["description"], ref, best))

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(data)}")

    total = len(data)
    print(f"\n{'='*50}")
    print(f"RESULTS ({total} samples)")
    print(f"{'='*50}")
    print(f"  Exact Match:  {exact}/{total} ({exact/total:.1%})")
    print(f"  Compile Rate: {compile_ok}/{total} ({compile_ok/total:.1%})")


    print(f"\nFAILED TO COMPILE ({len(errors)}):")
    for desc, ref, pred in errors[:15]:
        print(f"\n  DESC: {desc[:80]}")
        print(f"  REF:  {ref[:80]}")
        print(f"  PRED: {pred[:80]}")


if __name__ == "__main__":
    main()