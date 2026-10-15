from __future__ import annotations

import argparse
import json
import logging
import sys
import os
import re
from typing import Any, Dict, List
import boto3
from botocore.config import Config

def build_parser() -> argparse.ArgumentParser:
    """Day03のCLI引数を定義します（要件文とリトライ回数）。"""
    p = argparse.ArgumentParser(prog="day03")
    p.add_argument("--requirements", required=True)
    p.add_argument("--max-retry", type=int, default=1)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.requirements:
        raise ValueError("--requirements is required")
    if not (0 <= args.max_retry <= 3):
        raise ValueError("--max-retry must be between 0 and 3")

def _extract_json_text(raw: str) -> str:
    """LLM出力からコードブロックや前置きを除去し、{ ... } 部分だけを取り出す。"""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)   # 先頭の ```json を削除
    s = re.sub(r"\s*```$", "", s)            # 末尾の ``` を削除
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output does not contain a JSON object")
    return s[start : end + 1]

def _normalize(obj: Any) -> Dict[str, Any]:
    """中身まで検証し、仕様どおりの形に整える。問題があれば例外（→再生成）。"""
    if not isinstance(obj, dict):
        raise ValueError("top-level must be an object")

    title = obj.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    tasks = obj.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")

    norm_tasks = []
    for i, t in enumerate(tasks, start=1):
        if not isinstance(t, dict):
            raise ValueError(f"tasks[{i}] must be an object")
        desc = t.get("description")
        ac = t.get("acceptance_criteria")
        if not isinstance(desc, str) or not isinstance(ac, str):
            raise ValueError(f"tasks[{i}] needs description and acceptance_criteria")
        norm_tasks.append({"id": i, "description": desc, "acceptance_criteria": ac})

    risks = obj.get("risks", [])
    if not isinstance(risks, list) or not all(isinstance(r, str) for r in risks):
        raise ValueError("risks must be a list of strings")

    return {"title": title, "tasks": norm_tasks, "risks": risks}


SYSTEM_PROMPT = """あなたはソフトウェア開発の要件をタスクに分解するアシスタントです。
ユーザーから渡される要件を読み、次の形式のJSONオブジェクトを1つだけ出力してください。
前置き・説明文・コードブロック記号(```)は一切含めないでください。

{"title": string, "tasks": [{"id": integer, "description": string, "acceptance_criteria": string}], "risks": [string]}

ルール:
- キーは title, tasks, risks の3つのみ
- tasks は1〜10件、id は1からの連番
- acceptance_criteria は確認可能な完了条件を1文で書く
- risks は文字列の配列
- すべて日本語で書く"""


def _call_llm(requirements: str) -> str:
    """Bedrockを呼び出して生の出力を返す。DAY03_FAKE_LLM でテスト用の出力に差し替えられる。"""
    fake = os.getenv("DAY03_FAKE_LLM")
    if fake == "broken":
        return "はい、こちらがJSONです！ {title: 壊れてる"
    if fake == "ok":
        return '```json\n{"title": "テスト", "tasks": [{"id": 1, "description": "a", "acceptance_criteria": "b"}], "risks": []}\n```'

    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not region or not model_id:
        raise RuntimeError("AWS_REGION and BEDROCK_MODEL_ID must be set")

    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(connect_timeout=10, read_timeout=60),
    )
    response = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": f"要件:\n{requirements}"}]}],
        inferenceConfig={"temperature": 0.2, "maxTokens": 2000},
    )
    return "".join(
        block["text"]
        for block in response["output"]["message"]["content"]
        if "text" in block
    )


def generate_json(requirements: str) -> str:
    """要件文字列から、JSON文字列（本文のみ）を生成して返します。

    この関数を実装すると、`python -m day03.app --requirements ...` が動くようになります。

    実装ガイド：
    - LLMに「JSONだけを返す」ように強く指示する
    - `title` / `tasks` / `risks` を必ず含める
    - `tasks` は配列で、各要素に `id` / `description` / `acceptance_criteria` を含める
    - 返す文字列は JSON として `json.loads()` できる必要がある

    注意：
    - 余計な前置き/後置きの文章を混ぜない
    - 壊れやすいので、プロンプトは短く・形式を固定する
    """
    """
    # TODO(TRAINEE): Generate a JSON string that passes validate_json().
    """
    raw = _call_llm(requirements)
    try:
        obj = json.loads(_extract_json_text(raw))
        normalized = _normalize(obj)
    except Exception as e:
        logging.warning("invalid LLM output (%s): %.200s", e, raw)
        raise
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def validate_json(text: str) -> Dict[str, Any]:
    """生成結果のJSONを検証します（必須キーと型）。"""
    obj = json.loads(text)
    for key in ("title", "tasks", "risks"):
        if key not in obj:
            raise ValueError(f"missing key: {key}")
    if not isinstance(obj.get("tasks"), list):
        raise ValueError("tasks must be a list")
    return obj


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    JSON生成→検証→（失敗時は再生成）までを制御します。受講者は `generate_json()` を実装します。
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

    last_err: Exception | None = None
    for _ in range(args.max_retry + 1):
        try:
            text = generate_json(args.requirements)
            validate_json(text)
            print(text)
            return 0
        except NotImplementedError as e:
            logging.error(str(e))
            print(str(e), file=sys.stderr)
            return 1
        except Exception as e:
            last_err = e

    msg = str(last_err) if last_err else "validation failed"
    logging.error(msg)
    print(msg, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
