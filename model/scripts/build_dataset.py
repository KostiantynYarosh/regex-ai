import json
import argparse
import random
import re
import warnings
from pathlib import Path

PREFIXES = [
    "", "write a regex to ", "create regex for ", "give me regex to ",
    "I need a regex to ", "regex that can ", "pattern to ",
    "how to ", "help me ", "generate regex to ", "build regex for ",
    "find ", "match ", "extract ", "validate ", "detect ",
    "regex for ", "regular expression to ", "regexp to ", "where is ", 
    "where are ", "where",


    "i want to ", "i wanna ", "how do i ", "how can i ",
    "can you ", "could you ", "show me how to ",
    "what regex ", "what pattern ", "whats the regex for ",
    

    "parse ", "capture ", "check ", "verify ", "filter ",
    "search for ", "look for ", "grab ", "get ", "select ",
    "identify ", "recognize ", "spot ",
    

    "construct a regex to ", "define a pattern for ",
    "formulate regex to ", "compose a regex that ",
    "design a pattern to ", "provide regex for ",
    

    "in my text i want to ", "from a string ", "from text ",
    "in a string ",
]

SUFFIXES = [
    "", " please", " for me", " thanks", " thx", " pls",
    " in a string", " in text", " from input",
    " from a string", " if possible",
    " using regex", " with regex",
]

def augment_prompt(text: str) -> list[str]:
    variations = [text]
    for prefix in random.sample(PREFIXES, min(4, len(PREFIXES))):
        for suffix in random.sample(SUFFIXES, min(2, len(SUFFIXES))):
            aug = f"{prefix}{text}{suffix}".strip()
            if aug != text and len(aug) < 150:
                variations.append(aug)
    
    variations.append(text.lower())
    if text[0].islower():
        variations.append(text[0].upper() + text[1:])
    return list(set(variations))

def validate_regex(pattern: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            re.compile(pattern)
        return True
    except re.error:
        return False

def has_nonstandard_operators(pattern: str) -> bool:
    if re.search(r'\)\s*&\s*\(', pattern):
        return True
        return True
        return True
    return False

def escape_regex(pattern: str) -> str:
    return pattern.replace("\\", "<BS>")

def clean_description(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_txt_dataset(folder: Path) -> list[tuple[str, str]]:
    src_file = folder / "src.txt"
    targ_file = folder / "targ.txt"
    pairs = []
    
    if src_file.exists() and targ_file.exists():
        with open(src_file, 'r', encoding='utf-8') as f_src:
            descriptions = f_src.read().splitlines()
        with open(targ_file, 'r', encoding='utf-8') as f_targ:
            regexes = f_targ.read().splitlines()
        
        for desc, regex in zip(descriptions, regexes):
            desc = clean_description(desc)
            regex = regex.strip()
            if desc and regex:
                pairs.append((desc, regex))
    
    return pairs

def load_json_dataset(folder: Path) -> list[tuple[str, str]]:
    pairs = []
    
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                regex = item.get("regex", "").strip()
                if not regex or len(regex) > 60:
                    continue
                
                description = item.get("description", "").strip()
                if not description or len(description) < 5:
                    continue
                
                pairs.append((description, regex))
                
        except Exception as e:
            print(f"  Warning: Error reading {json_file.name}: {e}")
    
    return pairs

def filter_pair(desc: str, regex: str) -> tuple[bool, str]:
    if len(desc) < 3:
        return False, "description too short"
    if len(regex) < 1:
        return False, "regex empty"
    if len(regex) > 500:
        return False, "regex too long"
    if has_nonstandard_operators(regex):
        return False, "non-standard operators (~, &)"
    if not validate_regex(regex):
        return False, "invalid regex"
    
    return True, "ok"

def build_all(data_root: str):
    data_path = Path(data_root)
    datasets_dir = data_path / "datasets"
    processed_dir = data_path / "processed"
        
    all_pairs = []
    stats = {}
    
    print("=" * 70)
    print("Building Regex Dataset")
    print("=" * 70)
    
    for dataset_folder in sorted(datasets_dir.iterdir()):
        
        name = dataset_folder.name

        print(f"\n  Processing {name}...")
        
        pairs = load_txt_dataset(dataset_folder)
        pairs.extend(load_json_dataset(dataset_folder))
        
        if not pairs:
            print(f"     No valid data found")
            stats[name] = {"status": "empty", "raw": 0, "kept": 0}
            continue
        
        kept = []
        reasons = {}
        for desc, regex in pairs:
            ok, reason = filter_pair(desc, regex)
            if ok:
                kept.append((desc, regex))
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        
        stats[name] = {"status": "ok", "raw": len(pairs), "kept": len(kept)}
        all_pairs.extend(kept)
        
        print(f"     Raw: {len(pairs)} → Kept: {len(kept)}")
        if reasons:
            for reason, count in reasons.items():
                print(f"     Filtered: {count} ({reason})")
    
    seen_regex = set()
    unique_pairs = []
    for desc, regex in all_pairs:
        if regex not in seen_regex:
            seen_regex.add(regex)
            unique_pairs.append((desc, regex))
    
    print(f"\n{'=' * 70}")
    print(f"Total unique regex patterns: {len(unique_pairs)}")
    print(f"Duplicates removed: {len(all_pairs) - len(unique_pairs)}")
    
    dataset = []
    for desc, regex in unique_pairs:
        prompts = augment_prompt(desc)
        if len(prompts) > 6:
            prompts = random.sample(prompts, 6)
        
        escaped = escape_regex(regex)
        for p in prompts:
            dataset.append({
                "description": p,
                "regex": escaped
            })
    
    print(f"After prompt augmentation: {len(dataset)} training pairs")
    
    if len(dataset) == 0:
        print("No valid pairs found, exiting.")
        return
        
    random.seed(42)
    random.shuffle(dataset)
    
    n = len(dataset)
    train_end = int(n * 0.85)
    val_end = int(n * 0.95)
    
    splits = {
        "train": dataset[:train_end],
        "val": dataset[train_end:val_end],
        "test": dataset[val_end:]
    }
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print("Saving splits:")
    for name, data in splits.items():
        out_file = processed_dir / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}.json: {len(data):,} entries")

    print(f"\n{'=' * 70}")
    print("Dataset Summary:")
    print(f"{'Dataset':<20} {'Status':<10} {'Raw':>8} {'Kept':>8}")
    print("-" * 50)
    for name, s in sorted(stats.items()):
        print(f"{name:<20} {s['status']:<10} {s['raw']:>8} {s['kept']:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<20} {'':10} {sum(s['raw'] for s in stats.values()):>8} {sum(s['kept'] for s in stats.values()):>8}")

    print("\nBuild complete! Dataset is ready in data/processed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build clean regex training dataset")
    default_data_dir = str(Path(__file__).parent.parent / "data")
    parser.add_argument("--data_dir", default=default_data_dir, help="Path to data directory")
    
    args = parser.parse_args()
    build_all(args.data_dir)
