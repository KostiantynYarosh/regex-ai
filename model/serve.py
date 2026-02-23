

import regex as re
import argparse
import torch
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(title="Regex AI Model Server (v1.3.0)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
tokenizer = None
device = None

class GenerateRequest(BaseModel):
    description: str

class GenerateResponse(BaseModel):
    regex: str
    valid: bool
    tokens: list[dict]

class ValidateRequest(BaseModel):
    regex: str
    testText: str

class Match(BaseModel):
    text: str
    start: int
    end: int

class FeedbackRequest(BaseModel):
    description: str
    regex: str
    flag: str 

class ValidateResponse(BaseModel):
    valid: bool
    matches: list[Match]
    error: str | None = None

class TokenizeRequest(BaseModel):
    regex: str

class TokenizeResponse(BaseModel):
    tokens: list[dict]


TOKEN_COLORS = ["#3b82f6", "#6366f1", "#8b5cf6", "#64748b", "#0ea5e9", "#14b8a6", "#475569", "#4f46e5"]


REGEX_TOKEN_PATTERN = re.compile(
    r"""
    \[\^?(?:[^\]\\]|\\.)*\]          |
    \(\?(?::  |=  |!  |<=  |<!  |P<[a-zA-Z_]\w*>  |P=[a-zA-Z_]\w*) |
    \\[1-9]                           |
    \\[dwsbnrtfvDWSB.+*?^$|\\()\[\]{}] |
    \\x[0-9a-fA-F]{2}                |
    \\u[0-9a-fA-F]{4}                |
    \\.                               |
    \{\d+(?:,\d*)?\}                  |
    [*+?]\??                          |
    [()^$|]                           |
    .
    """,
    re.VERBOSE | re.DOTALL
)

LABELS = {
    r"\d": "digit (0-9)", r"\w": "word character", r"\s": "whitespace", r"\b": "word boundary",
    r"\D": "non-digit", r"\W": "non-word", r"\S": "non-whitespace", r"\.": "literal dot",
    r"\+": "literal plus", r"\-": "literal hyphen", r"\\": "literal backslash",
    "+": "one or more", "*": "zero or more", "?": "optional", "+?": "one or more (lazy)",
    "*?": "zero or more (lazy)", "^": "start of string", "$": "end of string", "|": "or",
    "(": "group start", ")": "group end", "(?:": "non-capturing group", "(?=": "positive lookahead",
    "(?!": "negative lookahead",
}


REGEX_TIMEOUT = 5

def safe_regex_match(pattern: str, text: str, timeout: int = REGEX_TIMEOUT):

    try:

        compiled = re.compile(pattern)
        matches = [
            {"text": m.group(), "start": m.start(), "end": m.end()}
            for m in compiled.finditer(text, timeout=timeout)
        ]
        return True, matches, None
    except TimeoutError:
        return False, [], f"Regex execution timed out after {timeout} seconds (possible ReDoS pattern)"
    except re.error as e:
        return False, [], str(e)
    except Exception as e:
        return False, [], str(e)


def unescape_regex(pattern: str) -> str:

    pattern = pattern.replace("<pad>", "").replace("</s>", "").replace("<unk>", "")

    pattern = pattern.replace("<BS> ", "\\").replace("<BS>", "\\")
    return pattern

def tokenize_regex(regex: str) -> list[dict]:
    raw_tokens = REGEX_TOKEN_PATTERN.findall(regex)
    result = []
    for i, token in enumerate(raw_tokens):
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        label = LABELS.get(token, f"literal '{token}'" if len(token) == 1 and token.isalnum() else f"token '{token}'")
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            label = f"NOT in: {inner[1:]}" if inner.startswith("^") else f"in: {inner}"
        elif token.startswith("{") and token.endswith("}"):
            inner = token[1:-1]
            label = f"exactly {inner}" if "," not in inner else (f"{inner.split(',')[0]} or more" if inner.split(',')[1] == "" else f"{inner.split(',')[0]}-{inner.split(',')[1]} times")
        result.append({"value": token, "label": label, "color": color})
    return result

@app.post("/api/tokenize", response_model=TokenizeResponse)
@limiter.limit("60/minute")
async def tokenize(req: TokenizeRequest, request: Request):
    return TokenizeResponse(tokens=tokenize_regex(req.regex))

@app.post("/api/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
async def generate(req: GenerateRequest, request: Request):
    input_text = f"generate regex: {req.description}"
    ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)
    with torch.no_grad():
        outs = model.generate(ids, max_length=128, num_beams=4, num_return_sequences=4, early_stopping=True)
    

    bs_token_id = tokenizer.convert_tokens_to_ids("<BS>")
    skip_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id}
    skip_ids.discard(bs_token_id)
    
    best, valid = "", False
    for out in outs:

        filtered = [t for t in out.tolist() if t not in skip_ids]
        cand = tokenizer.decode(filtered, skip_special_tokens=False)
        cand = unescape_regex(cand).strip()
        try:
            re.compile(cand)
            best, valid = cand, True
            break
        except re.error:
            if not best: best = cand
    return GenerateResponse(regex=best, valid=valid, tokens=tokenize_regex(best))

@app.post("/api/validate", response_model=ValidateResponse)
@limiter.limit("30/minute")
async def validate(req: ValidateRequest, request: Request):
    valid, matches, error = safe_regex_match(req.regex, req.testText)
    match_objects = [Match(**m) for m in matches] if valid else []
    return ValidateResponse(valid=valid, matches=match_objects, error=error)

@app.post("/api/feedback")
@limiter.limit("30/minute")
async def feedback(req: FeedbackRequest, request: Request):
    from datetime import datetime
    import json
    import os
    
    feedback_file = Path(__file__).parent / "data" / "feedback.json"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "description": req.description,
        "regex": req.regex,
        "flag": req.flag,
        "timestamp": int(datetime.now().timestamp())
    }
    
    # Read existing data if it exists
    data = []
    if feedback_file.exists():
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # Start fresh if file is corrupted

    data.append(entry)
    
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return {"status": "ok"}

@app.get("/api/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {"status": "ok", "version": "1.3.0", "endpoints": ["/api/generate", "/api/validate", "/api/tokenize", "/api/feedback"]}

def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=str(Path(__file__).parent / "checkpoints" / "t5-regex"))
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<BS>", "{", "}", "^", "<", "`", "~"]})
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device).eval()
    print(f"Registered routes: {[r.path for r in app.routes]}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
