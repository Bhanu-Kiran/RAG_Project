#!/usr/bin/env python3
"""
extract_and_summarize.py

Usage:
    python extract_and_summarize.py /path/to/notebooks_dir --out ./summaries

Requirements:
    pip install nbformat openai

Environment:
    export OPENAI_API_KEY="sk-..."

What it does (short):
 - Parses .ipynb files
 - Extracts imports and top-level functions/classes via AST
 - Heuristically picks key functions (docstring, calls, length)
 - Calls OpenAI to produce a concise JSON summary per notebook:
   {"title","summary","imports":[{"name","purpose"}],"key_functions":[{"name","takeaway"}]}
 - Saves results to out_dir/<notebook_name>.summary.json
"""

import os
import sys
import json
import argparse
import nbformat
import ast
import re
from collections import Counter, defaultdict
from textwrap import shorten
import openai
from typing import List, Dict, Tuple
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Add it to .env or environment.")

# -------------------------------------------------------
# OpenAI v1 Client
# -------------------------------------------------------
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------------------------------------------------------
# Config
# -------------------------------------------------------
MAX_SNIPPET_LINES = 20
TOP_K_FUNCTIONS = 3
MAX_SUMMARY_TOKENS = 350

IMPORT_RE = re.compile(r'^\s*(?:from\s+([^\s]+)\s+import|import\s+([^\s,]+))', re.MULTILINE)

def list_ipynb_files(root_dir: str) -> List[str]:
    files = []
    for p in os.listdir(root_dir):
        if p.endswith(".ipynb"):
            files.append(os.path.join(root_dir, p))
    return sorted(files)

def read_notebook(path: str):
    nb = nbformat.read(path, as_version=4)
    return nb

def get_cells(nb) -> List[dict]:
    return nb.get("cells", [])

# AST helpers
def extract_imports_from_code(code: str) -> List[Tuple[str,str]]:
    """
    Returns list of tuples (module, imported_name_or_alias)
    e.g., import numpy as np -> ('numpy','np'), from sklearn.model_selection import train_test_split -> ('sklearn.model_selection','train_test_split')
    """
    results = []
    try:
        tree = ast.parse(code)
    except Exception:
        # fallback to regex if code is messy
        for m in IMPORT_RE.finditer(code):
            mod = m.group(1) or m.group(2)
            alias = None
            results.append((mod, alias))
        return results

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                results.append((n.name, n.asname or n.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for n in node.names:
                name = n.name
                alias = n.asname or name
                results.append((module, alias))
    return results

def extract_top_level_defs(code: str) -> List[Dict]:
    """
    Returns list of dicts: {name, type('function'/'class'), lineno, end_lineno, docstring, source_lines}
    """
    defs = []
    try:
        tree = ast.parse(code)
    except Exception:
        return defs

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node)
            # get source lines (approx)
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)  # python3.8+
            src_lines = code.splitlines()[start:end]
            defs.append({"name": node.name, "kind": "function", "lineno": node.lineno,
                         "end_lineno": end, "doc": doc, "source": "\n".join(src_lines)})
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            src_lines = code.splitlines()[start:end]
            defs.append({"name": node.name, "kind": "class", "lineno": node.lineno,
                         "end_lineno": end, "doc": doc, "source": "\n".join(src_lines)})
    return defs

def snippet_for_provenance(cell_src: List[str], max_lines=MAX_SNIPPET_LINES) -> str:
    text = "".join(cell_src) if isinstance(cell_src, list) else str(cell_src)
    lines = text.splitlines()
    return "\n".join(lines[:max_lines])

def score_function(def_info: Dict, full_code_text: str) -> float:
    """
    Heuristic score for selecting 'key' functions:
      - docstring presence: +2
      - length (lines): + len/50
      - number of references/calls found in the full notebook code: +2 * call_count
    """
    score = 0.0
    if def_info.get("doc"):
        score += 2.0
    # length
    src = def_info.get("source","")
    nlines = len(src.splitlines())
    score += nlines / 50.0
    # count calls by name in full_code_text (word-boundary)
    name = def_info.get("name")
    if name:
        occurrences = len(re.findall(r"\b{}\b".format(re.escape(name)), full_code_text))
        # subtract the definition itself
        occurrences = max(0, occurrences - 1)
        score += 2.0 * occurrences
    return score

def normalize_imports(import_tuples: List[Tuple[str,str]]) -> Dict[str, Dict]:
    """
    Aggregate imports into mapping: module -> {'aliases':set(), 'count':int}
    """
    agg = {}
    for mod, alias in import_tuples:
        if not mod:
            continue
        key = mod.strip()
        if key not in agg:
            agg[key] = {"aliases": set(), "count": 0}
        if alias:
            agg[key]["aliases"].add(alias)
        agg[key]["count"] += 1
    # convert aliases to list
    for k in list(agg.keys()):
        agg[k]["aliases"] = sorted(list(agg[k]["aliases"]))
    return agg

# Prompt builder
SYSTEM_PROMPT = """You are a concise summarizer. Using ONLY the EVIDENCE block provided, produce a JSON object with these keys:
 - title: one-line title
 - summary: 3-6 short lines describing notebook purpose and main process (concise)
 - imports: list of {"name":PACKAGE,"purpose":ONE_WORD_OR_SHORT_PHRASE}
 - key_functions: list of {"name":FUNC_NAME,"takeaway":ONE_LINE_PURPOSE}
Do NOT add anything not supported by the evidence. Dont process the custom functions as main functions... Focus on the imported functions before processing custom functions.
If uncertain, mark the takeaway with "UNCERTAIN".
Return JSON only. Keep everything short.
"""

USER_PROMPT_TEMPLATE = """
===EVIDENCE_START===
{evidence}
===EVIDENCE_END===
Instructions: Use only the evidence. Provide concise JSON as specified in the system prompt.
"""

def build_evidence_block(nb_path: str, first_md: str, sample_code_snippets: List[Tuple[int,str]],
                         imports_agg: Dict, key_defs: List[Dict]) -> str:
    pieces = []
    pieces.append(f"[file: {os.path.basename(nb_path)}]")
    if first_md:
        pieces.append("[first_markdown]")
        pieces.append(first_md.strip())
    pieces.append("[imports]")
    # limited import lines
    for m, info in list(imports_agg.items())[:50]:
        aliases = ",".join(info.get("aliases",[])) or ""
        pieces.append(f"{m}  # aliases={aliases}  # count={info.get('count')}")
    pieces.append("[key_defs]")
    for d in key_defs:
        # include function signature line and first few lines
        src = d.get("source","")
        snippet = "\n".join(src.splitlines()[:6])
        pieces.append(f"def/class: {d.get('name')} (type={d.get('kind')})\n{shorten(snippet, 400)}")
    pieces.append("[samples]")
    for idx, s in sample_code_snippets:
        pieces.append(f"[cell {idx}]\n{shorten(s, 800)}")
    return "\n\n".join(pieces)

def call_openai_chat(system: str, user: str, model=OPENAI_MODEL, max_tokens=MAX_SUMMARY_TOKENS) -> str:
    """
    Uses the new openai v1 client (OpenAI) to create a chat completion.
    """
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content

def summarize_notebook(nb_path: str) -> Dict:
    nb = read_notebook(nb_path)
    cells = get_cells(nb)
    # collect first markdown cell (if any)
    first_md = ""
    for c in cells:
        if c.get("cell_type") == "markdown" and c.get("source", "").strip():
            first_md = c.get("source")
            break

    code_text_concat = ""
    imports_all = []
    defs_all = []
    sample_code = []

    for idx, c in enumerate(cells):
        ctype = c.get("cell_type")
        src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        if ctype == "code":
            code_text_concat += "\n" + src
            # imports from this cell
            imports_all.extend(extract_imports_from_code(src))
            # top-level defs
            defs_all.extend(extract_top_level_defs(src))
            # capture a few code snippets for evidence
            if len(sample_code) < 6 and src.strip():
                sample_code.append((idx, snippet_for_provenance(src.splitlines(), max_lines=20)))
        elif ctype == "markdown":
            # optional: scan for inline mentions of packages (sketch)
            m = IMPORT_RE.search(src)
            if m:
                # rough capture
                imports_all.append((m.group(1) or m.group(2), None))

    imports_agg = normalize_imports(imports_all)
    # compute function scores
    scored = []
    for d in defs_all:
        sc = score_function(d, code_text_concat)
        d["_score"] = sc
        scored.append(d)
    # pick top-K by score
    scored_sorted = sorted(scored, key=lambda x: x.get("_score",0), reverse=True)
    key_defs = scored_sorted[:TOP_K_FUNCTIONS]

    # build evidence block
    evidence = build_evidence_block(nb_path, first_md, sample_code, imports_agg, key_defs)

    # call LLM
    user_prompt = USER_PROMPT_TEMPLATE.format(evidence=evidence)
    try:
        model_out = call_openai_chat(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

    # parse JSON from model output (best-effort)
    parsed = {}
    try:
        parsed = json.loads(model_out)
    except Exception:
        # try to extract JSON substring
        jmatch = re.search(r'(\{.*\})', model_out, re.S)
        if jmatch:
            try:
                parsed = json.loads(jmatch.group(1))
            except Exception:
                parsed = {"raw": model_out}
        else:
            parsed = {"raw": model_out}

    # attach deterministic extras
    parsed["_deterministic"] = {
        "imports": imports_agg,
        "key_defs_raw": [{k: v for k, v in d.items() if k in ("name","kind","doc","lineno","_score")} for d in key_defs]
    }
    return parsed

# ----------------- REPLACE/ADD: Enhanced Import-usage analyzer -----------------
def analyze_import_usage(imports_agg: dict, full_code: str, sample_snippets: List[tuple]) -> dict:
    """
    For each module in imports_agg, return:
      - aliases: list
      - count: import occurrences
      - evidence: up to 3 real code lines where module/alias is used
      - inferred_purpose: short hint (single phrase)
      - detailed_purpose: 1-2 sentence deterministic description of how it's used in these notebooks
      - example_usage: short example line (derived from evidence or heuristic)
    """
    usage = {}

    # richer mapping: short purpose -> detailed purpose sentence
    detailed_map = {
        "pandas": ("Dataframes and CSV/Excel I/O. Used to load, inspect, and transform tabular data "
                   "via DataFrame operations like read_csv, groupby, merge."),
        "numpy": ("Numeric arrays and vectorized computation. Used for numeric transforms, array ops, "
                  "and as backend for numeric computations."),
        "sklearn": ("Classic ML utilities (preprocessing, model selection, metrics). Used for train_test_split, "
                    "pipelines, and evaluation metrics."),
        "xgboost": ("Gradient-boosted decision trees for modeling. Used to train and evaluate XGBoost models."),
        "torch": ("PyTorch deep learning framework. Used to build/train neural networks and handle tensors."),
        "langchain_text_splitters": ("Text chunking utilities. Used to split long documents into chunks for embeddings."),
        "langchain_core.documents": ("Document abstraction layer. Used to wrap text and metadata into Document objects."),
        "langchain_community.document_loaders": ("File/dir document loaders. Used to load data from directories, PDFs, or text files."),
        "openai": ("OpenAI API client for embeddings and LLM calls. Used to request embeddings or chat completions."),
        "chromadb": ("Vector DB client. Used to persist and query vector embeddings."),
    }

    # helper: flatten code into lines
    code_lines = [ln for ln in full_code.splitlines() if ln.strip()]

    def find_evidence(tokens):
        found = []
        for ln in code_lines:
            for t in tokens:
                # match patterns like "alias." or "Module(" or "module.function"
                if re.search(rf"(\b{re.escape(t)}\b\.)|(\b{re.escape(t)}\b\()", ln):
                    found.append(ln.strip())
                    break
            if len(found) >= 3:
                break
        return found

    for mod, info in imports_agg.items():
        aliases = info.get("aliases", []) or []
        tokens = []
        tokens.extend(aliases)
        last = mod.split(".")[-1]
        tokens.append(last)
        # also try the top-level module token (first part)
        first = mod.split(".")[0]
        if first and first != last:
            tokens.append(first)

        evidence = find_evidence(tokens)
        # if none found in full code, inspect sample snippets
        if not evidence and sample_snippets:
            for _, s in sample_snippets:
                for t in tokens:
                    if re.search(rf"\b{re.escape(t)}\b", s):
                        evidence.append(s.strip())
                        break
                if len(evidence) >= 3:
                    break

        # example usage: prefer a short cleaned evidence line or a synthesized one
        example_usage = None
        if evidence:
            example_usage = evidence[0]
            # shorten long lines to ~120 chars
            if len(example_usage) > 140:
                example_usage = example_usage[:140].rsplit(" ", 1)[0] + "..."
        else:
            # fallback heuristics
            if "split" in mod or any("split" in a.lower() for a in aliases):
                example_usage = "splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)"
            elif "loader" in mod or "load" in mod:
                example_usage = "docs = DirectoryLoader('data/').load()"
            else:
                example_usage = f"# usage of {mod} not directly found; inspect code"

        # inferred purpose short (one phrase)
        inferred_short = detailed_map.get(mod, None)
        if inferred_short:
            inferred_purpose = inferred_short.split(".")[0]  # first clause as short phrase
            detailed_purpose = inferred_short
        else:
            # simple heuristics fallback
            if "split" in mod or any("split" in a.lower() for a in aliases):
                inferred_purpose = "text chunking/splitting"
                detailed_purpose = "Utilities to split long text into manageable chunks for embedding and retrieval."
            elif "loader" in mod or "load" in mod:
                inferred_purpose = "document loading"
                detailed_purpose = "Load files or directories into document objects for processing."
            elif "text" in mod:
                inferred_purpose = "text utilities"
                detailed_purpose = "Helpers for text manipulation and processing."
            else:
                inferred_purpose = "utility (inspect usage)"
                detailed_purpose = f"Module {mod}. Inspect evidence lines to determine exact role."

        usage[mod] = {
            "aliases": aliases,
            "count": info.get("count", 0),
            "evidence": evidence,
            "inferred_purpose": inferred_purpose,
            "detailed_purpose": detailed_purpose,
            "example_usage": example_usage
        }

    return usage
# ----------------- END REPLACE/ADD -----------------


def main():
    parser = argparse.ArgumentParser(description="Extract and summarize notebooks (short outputs).")
    parser.add_argument("notebooks_dir", help="Directory containing .ipynb files")
    parser.add_argument("--out", "-o", default="./nb_summaries", help="Output directory for summaries")
    args = parser.parse_args()

    nb_dir = args.notebooks_dir
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    files = list_ipynb_files(nb_dir)
    if not files:
        print("No .ipynb files found in:", nb_dir)
        sys.exit(1)

    print(f"Found {len(files)} notebooks. Processing (this will call OpenAI per notebook).")
    for nb_path in files:
        print(f" - {os.path.basename(nb_path)} ...", end="", flush=True)
        try:
            summary = summarize_notebook(nb_path)
            out_file = os.path.join(out_dir, os.path.basename(nb_path) + ".summary.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(" saved ->", out_file)
        except Exception as e:
            print(" ERROR:", str(e))

if __name__ == "__main__":
    main()


## Code to parse the .ipynb files and create a tree view of the files in the directory
# import os

# def print_ipynb_tree(root_folder):
#     for current_path, dirs, files in os.walk(root_folder):
#         # Filter only .ipynb files
#         ipynb_files = [f for f in files if f.endswith(".ipynb")]
#         if not ipynb_files:
#             continue

#         # Calculate nesting depth
#         level = current_path.replace(root_folder, "").count(os.sep)
#         indent = "    " * level
#         folder_name = os.path.basename(current_path)

#         print(f"{indent}|-- {folder_name}/")
#         for f in ipynb_files:
#             print(f"{indent}    |-- {f}")

# # Usage:
# print_ipynb_tree(r"C:\Data_Science\RAG\Project")   # change to your folder
