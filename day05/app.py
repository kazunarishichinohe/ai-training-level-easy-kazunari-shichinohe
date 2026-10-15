from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import unicodedata  
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from day02.app import invoke_bedrock


def build_parser() -> argparse.ArgumentParser:
    """Day05のCLI引数を定義します（質問文）。"""
    p = argparse.ArgumentParser(prog="day05")
    p.add_argument("--question", required=True)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.question or not args.question.strip():
        raise ValueError("--question is required")

# ===== データ読み込み・分割 =====

DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNK_SIZE = 400  # 1チャンクあたりの最大文字数


@dataclass
class Chunk:
    """検索の単位。どのファイルの、どの部分かを持つ。"""

    source: str  # 例: day05/data/検索拡張生成.txt
    title: str   # 例: 検索拡張生成（ファイル名から作る）
    text: str


def _split_text(text: str, chunk_size: int) -> List[str]:
    """行（段落）単位でまとめて、chunk_size 文字以内のかたまりに分ける。"""
    lines = [ln.strip() for ln in text.splitlines()]
    # 空行と、Wikipediaの見出し行（"== 概要 ==" など）は除く
    lines = [ln for ln in lines if ln and not ln.startswith("==")]

    chunks: List[str] = []
    buf = ""
    for ln in lines:
        # 1段落だけで長すぎる場合は、chunk_size ごとに切る
        while len(ln) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(ln[:chunk_size])
            ln = ln[chunk_size:]
        # 足すと長すぎるなら、いったん区切る
        if buf and len(buf) + 1 + len(ln) > chunk_size:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{ln}" if buf else ln
    if buf:
        chunks.append(buf)
    return chunks


def load_chunks(data_dir: Path = DATA_DIR, chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
    """data_dir 配下の .txt を読み込み、チャンクのリストにする。

    読み込めない場合は RuntimeError（main で exit code=1 になる）。
    """
    if not data_dir.is_dir():
        raise RuntimeError(f"データフォルダが見つかりません: {data_dir}")
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise RuntimeError(f".txt ファイルがありません: {data_dir}")

    chunks: List[Chunk] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise RuntimeError(f"読み込みに失敗しました: {path.name}: {e}") from e
        source = f"day05/data/{path.name}"
        title = path.stem.replace("_", " ")
        for body in _split_text(text, chunk_size):
            chunks.append(Chunk(source=source, title=title, text=body))
    return chunks

# ===== 検索（文字bigram） =====

TOP_K = 3          # 回答に使うチャンクの最大数
MIN_SCORE = 0.5    # これ未満は「ヒットなし」扱い

# 質問によく出るが、内容とは関係ない言い回し
_STOPWORDS = ["について", "教えて", "ください", "何ですか", "とは"]
_HIRAGANA_ONLY = re.compile(r"^[\u3040-\u309f]+$")
_SEPARATORS = re.compile(r"[\s\W_]+")


def _bigrams(text: str) -> set[str]:
    """文字列を「2文字ずつ」の集合にする（例: 検索拡張 → 検索, 索拡, 拡張）。"""
    text = unicodedata.normalize("NFKC", text).lower()
    result: set[str] = set()
    for seg in _SEPARATORS.split(text):  # 空白・記号で区切ってから作る
        result.update(seg[i : i + 2] for i in range(len(seg) - 1))
    return result


def _query_bigrams(question: str) -> set[str]:
    """質問のbigram。言い回しと、ひらがなだけのもの（の・は等）は除く。"""
    for w in _STOPWORDS:
        question = question.replace(w, " ")
    return {b for b in _bigrams(question) if not _HIRAGANA_ONLY.match(b)}


def search(
    question: str,
    chunks: List[Chunk],
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> List[Tuple[float, Chunk]]:
    """質問のbigramがどれだけチャンクに含まれるか（0〜1）で順位付けする。"""
    q = _query_bigrams(question)
    if not q:
        return []
    scored: List[Tuple[float, Chunk]] = []
    for c in chunks:
        score = len(q & _bigrams(f"{c.title} {c.text}")) / len(q)
        if score >= min_score:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ===== 回答生成（Bedrock） =====

NO_ANSWER = "NO_ANSWER"  # 資料から答えられないときに、LLMに返させる目印


def _build_prompt(question: str, hits: List[Tuple[float, Chunk]]) -> str:
    """検索で見つかったチャンクを資料として埋め込んだプロンプトを作る。"""
    context = "\n\n".join(
        f"[{i}] 出典: {c.source}\n{c.text}" for i, (_, c) in enumerate(hits, 1)
    )
    return f"""あなたは資料に基づいて回答するアシスタントです。
次の<資料>だけを根拠に、<質問>に日本語で簡潔に（3文以内で）答えてください。
- 資料に書かれていないことを、推測で補わないでください。
- 資料から答えられない場合は、{NO_ANSWER} とだけ出力してください。
- 前置き・見出し・出典番号は書かず、回答の本文だけを出力してください。

<資料>
{context}
</資料>

<質問>
{question}
</質問>"""


def generate_answer(question: str, hits: List[Tuple[float, Chunk]]) -> Optional[str]:
    """Bedrockで回答を作る。資料から答えられない場合は None を返す。"""
    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not region or not model_id:
        raise RuntimeError("AWS_REGION と BEDROCK_MODEL_ID を設定してください")

    reply = invoke_bedrock(
        prompt=_build_prompt(question, hits),
        region=region,
        model_id=model_id,
        temperature=0.0,
        max_tokens=512,
        timeout_sec=30,
    ).strip()

    if NO_ANSWER in reply:
        return None
    return " ".join(reply.split())  # 改行をまとめて1行にする

def answer_with_rag(question: str) -> str:
    """RAGで質問に回答し、指定フォーマットのテキストを返します。

    この関数を実装すると、`python -m day05.app --question ...` が動くようになります。

    要件（READMEの出力フォーマット）：
    - 標準出力に次の形で出すための文字列を返す
      1) `Answer:` 行
      2) `Sources:` 行
      3) `- <URL or ファイル名> (excerpt: "...")` を最低1件（ヒットなしなら `- (none)`）

    実装ガイド：
    - `day05/data/` 配下の `.txt` を読み込み、検索対象とする
    - 最初は単純なキーワード検索でもOK（高品質でなくてよい）
    - ヒットがない場合の挙動を必ず実装する
    """
# ===== 出力の組み立て =====

EXCERPT_CHARS = 80          # 出典に表示する抜粋の最大文字数
MAX_QUESTION_CHARS = 1000   # これより長い質問は切り詰める


def _excerpt(text: str) -> str:
    """抜粋を1行・EXCERPT_CHARS 文字以内にする（" は ' に置き換える）。"""
    one_line = " ".join(text.split()).replace('"', "'")
    if len(one_line) <= EXCERPT_CHARS:
        return one_line
    return one_line[:EXCERPT_CHARS] + "…"


def _format_no_hit(chunks: List[Chunk]) -> str:
    """ヒットなしのときの出力。次に何をすればよいかが分かるようにする。"""
    titles = "、".join(sorted({c.title for c in chunks}))
    return (
        "Answer: 該当する根拠が見つかりませんでした。"
        f"対象データ（{titles}）に関する質問に言い換えるか、"
        "day05/data/ に関連する .txt を追加してから再実行してください。\n"
        "Sources:\n"
        "- (none)"
    )


def answer_with_rag(question: str) -> str:
    """RAGで質問に回答し、指定フォーマットのテキストを返します。

    1. day05/data/ の .txt を読み込み、チャンクに分ける
    2. 質問に近いチャンクを検索する（文字bigram）
    3. ヒットしたチャンクを資料としてBedrockに渡し、回答を作る
    4. Answer / Sources の形式で返す（根拠がなければ Sources は - (none)）
    """
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        logging.warning("質問が長いため、先頭 %d 文字だけを使います", MAX_QUESTION_CHARS)
        question = question[:MAX_QUESTION_CHARS]

    chunks = load_chunks()
    hits = search(question, chunks)
    if not hits:
        return _format_no_hit(chunks)

    answer = generate_answer(question, hits)
    if answer is None:  # 検索ではヒットしたが、内容としては答えられなかった
        return _format_no_hit(chunks)

    lines = [f"Answer: {answer}", "Sources:"]
    lines += [f'- {c.source} (excerpt: "{_excerpt(c.text)}")' for _, c in hits]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `answer_with_rag()` を実装します。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_args(args)
    except Exception as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2

    try:
        out = answer_with_rag(args.question)
        print(out)
        return 0
    except NotImplementedError as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        logging.error("%s", e)
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
