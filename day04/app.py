from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
logger = logging.getLogger("day04")


def build_parser() -> argparse.ArgumentParser:
    """Day04のCLI引数を定義します（ユーザー入力テキスト）。"""
    p = argparse.ArgumentParser(prog="day04")
    p.add_argument("--text", required=True)
    return p


def _validate_args(args: argparse.Namespace) -> None:
    """引数の簡易バリデーションを行います（入力不備は exit code=2）。"""
    if not args.text:
        raise ValueError("--text is required")


# ===== ツール定義 =====

# LLMが呼び出してよいツールの許可リスト
ALLOWED_TOOLS = {"add"}

# 引数として受け付ける整数の範囲（極端に大きな値を弾く）
MAX_ABS_VALUE = 10**12


@tool
def add(a: int, b: int) -> int:
    """2つの整数 a と b を足し算して、その結果を返す。"""
    logger.info("TOOL CALLED: add(a=%s, b=%s)", a, b)
    return a + b


def _validate_add_args(args: Any) -> Dict[str, int]:
    """add の引数を検証する。不正なら ValueError を投げ、ツールは実行しない。"""
    if not isinstance(args, dict):
        raise ValueError(f"add: 引数の形式が不正です: {args!r}")
    if set(args.keys()) != {"a", "b"}:
        raise ValueError(f"add: 引数は a と b の2つだけです: {sorted(args.keys())}")

    for name in ("a", "b"):
        value = args[name]
        # bool は int のサブクラスなので、明示的に除外する
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"add: {name} は整数である必要があります: {value!r}")
        if abs(value) > MAX_ABS_VALUE:
            raise ValueError(f"add: {name} が大きすぎます（上限 {MAX_ABS_VALUE}）: {value}")

    return {"a": args["a"], "b": args["b"]}


def _execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """LLMからのツール呼び出し要求を検証してから実行する。"""
    name = tool_call.get("name")
    args = tool_call.get("args")

    if name not in ALLOWED_TOOLS:
        logger.warning("TOOL REJECTED: 許可されていないツールです: %s", name)
        raise ValueError(f"許可されていないツールです: {name}")

    try:
        valid_args = _validate_add_args(args)
    except ValueError:
        logger.warning("TOOL REJECTED: add の引数が不正です: %r", args)
        raise

    result = add.invoke(valid_args)
    return str(result)

def run_chain(text: str) -> str:
    """LangChain + Tool calling を使って回答（文字列）を返します。

    この関数を実装すると、`python -m day04.app --text ...` が動くようになります。

    要件（READMEの受け入れ基準）：
    - `today` または `add` のツールを1つ実装し、LLMから1回以上呼び出す
    - ツール引数のバリデーションを入れる（不正なら実行しない）
    - ツール失敗時は安全に失敗する（例外でOK。mainがexit code=1にする）

    ヒント：
    - まずはツールをPython関数として作り、ログで「呼ばれた」ことを確認
    - 次にLLM側のプロンプトで「必要ならツールを使う」よう誘導
    """
    # ===== チェーン定義 =====

SYSTEM_PROMPT = """あなたは計算アシスタントです。
足し算が必要なときは、暗算せずに必ず add ツールを使ってください。
ツールの結果をもとに、日本語で1〜2文の簡潔な最終回答を書いてください。
回答はプレーンテキストで書き、マークダウン記法（**など）は使わないでください。"""


# 入力の最大文字数（長すぎる入力でコストや処理時間が膨らむのを防ぐ）
MAX_INPUT_CHARS = 2000

# LLMとツールの往復回数の上限（無限ループ防止）
MAX_TOOL_ROUNDS = 5


def _build_chain():
    """プロンプト → ツール付きLLM の順につないだチェーンを作る。"""
    region = os.getenv("AWS_REGION")
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not region or not model_id:
        raise RuntimeError("AWS_REGION と BEDROCK_MODEL_ID を設定してください")

    llm = ChatBedrockConverse(
        model=model_id,
        region_name=region,
        temperature=0,
        max_tokens=1024,
    )
    llm_with_tools = llm.bind_tools([add])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{text}"),
        MessagesPlaceholder("scratchpad", optional=True),
    ])

    return prompt | llm_with_tools


def _content_to_text(content: Any) -> str:
    """LLMの返答（文字列 or ブロックのリスト）から本文テキストだけを取り出す。"""
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts).strip()


def run_chain(text: str) -> str:
    """LangChain + Tool calling を使って回答（文字列）を返します。"""
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"入力が長すぎます（{len(text)}文字 / 上限{MAX_INPUT_CHARS}文字）")

    chain = _build_chain()
    scratchpad: List[Any] = []  # LLMの返答とツール結果をためておく場所
    tool_call_count = 0

    for round_no in range(1, MAX_TOOL_ROUNDS + 1):
        ai_msg = chain.invoke({"text": text, "scratchpad": scratchpad})
        scratchpad.append(ai_msg)

        # ツール呼び出しがなければ、それが最終回答
        if not ai_msg.tool_calls:
            if tool_call_count == 0:
                logger.warning("ツールが一度も呼ばれずに回答が返されました")
            answer = _content_to_text(ai_msg.content)
            if not answer:
                raise RuntimeError("LLMから空の回答が返されました")
            return answer

        # ツール呼び出しがあれば、検証してから実行し、結果をLLMに返す
        for tool_call in ai_msg.tool_calls:
            logger.info(
                "LLM REQUESTED (round %d): %s %s",
                round_no, tool_call.get("name"), tool_call.get("args"),
            )
            result = _execute_tool_call(tool_call)  # 不正なら例外 → exit code 1
            tool_call_count += 1
            scratchpad.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    raise RuntimeError(f"ツール呼び出しが上限回数（{MAX_TOOL_ROUNDS}回）を超えました")


def main(argv: List[str] | None = None) -> int:
    """CLIのエントリポイントです。

    受講者は `run_chain()` の実装に集中し、ここは原則編集しません。
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
        out = run_chain(args.text)
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
