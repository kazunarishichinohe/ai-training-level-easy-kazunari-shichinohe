# Day 05

## 受講者が実装する場所

- `day05/app.py` の `answer_with_rag()`

## 目的

- 公開データを使ってRAGの最小構成を作る
- 回答に根拠（出典）を必ず付ける

## ゴール（完了条件）

- `Answer:` と `Sources:` が必ず出力される
- ヒットなしの場合でも `Sources:` に `- (none)` が出る

## データ出典/ライセンス

- データ名：
日本語版Wikipedia「大規模言語モデル」「検索拡張生成」「Amazon Web Services」の記事本文（2026-10-15 に MediaWiki API の extracts でプレーンテキストとして取得し、day05/data/ に記事ごとの .txt で保存）
- URL：
https://ja.wikipedia.org/wiki/大規模言語モデル 、https://ja.wikipedia.org/wiki/検索拡張生成 、https://ja.wikipedia.org/wiki/Amazon_Web_Services
- ライセンス：
クリエイティブ・コモンズ 表示-継承 4.0 国際（CC BY-SA 4.0） https://creativecommons.org/licenses/by-sa/4.0/deed.ja

## 機能要件

コマンド

- `python -m day05.app` で起動できること（モジュール名は固定）

データ配置

- `day05/data/` 配下に、取り込むドキュメントを配置する
  - 形式はテキスト（`.txt`）でOK
  - 公開データから取得した本文を貼り付けてもよい

必須オプション

- `--question`：質問（文字列、必須）

出力（標準出力）

以下の形式で出力してください。

1. `Answer:` 行に回答
2. `Sources:` の下に参照元を列挙（最低1件）

例：

```
Answer: ...
Sources:
- <URL or ファイル名> (excerpt: "...")
```

検索ヒットなし時

- `Answer:` は「該当する根拠が見つからない」旨を返す
- `Sources:` は空にせず、代わりに `- (none)` を出す

終了コード

- 正常終了：`0`
- 入力不備：`2`
- 実行失敗（データ読み込み失敗等）：`1`

## 実装タスク

- `day05/app.py` の `answer_with_rag()` を実装し、機能要件を満たす
- `day05/data/` にデータを配置し、出典/ライセンスを README に記載する

## 実行方法

### データの取得/配置方法

1. 公開データ（テキスト形式）を用意する
2. `day05/data/` ディレクトリを作成し、テキストファイルを配置する

```bash
mkdir -p day05/data
# 例：公開データをテキストで保存
# ※ 出典とライセンスを上記「データ出典/ライセンス」セクションに記載すること
```

Windows（PowerShell）の場合：

```powershell
New-Item -ItemType Directory -Force -Path day05\data
```

### 実行コマンド例

```bash
# 基本的な実行
python -m day05.app --question "〇〇について教えて"
```

### 期待する出力例

```
Answer: 〇〇は...です。

Sources:
- day05/data/sample.txt (excerpt: "〇〇は...")
```

---

**以下は受講者が記入してください**

## 提出物

- `day05/app.py`
- `day05/README.md`

## セルフレビュー

- 正常系：答えと根拠がセットで出る
「AWSはいつサービスを開始しましたか」で「2006年」と回答し、Sources に出典3件（ファイル名と抜粋）が出ることを確認。exit=0
- 異常系：検索ヒットがない場合に、次の行動が分かるメッセージになる
「明日の東京の天気は？」で、対象データの一覧と「質問の言い換え／.txt の追加」を案内し、Sources は `- (none)`。exit=0。--question なし・空白のみは exit=2、day05/data がない場合は exit=1 を確認
- 境界：質問が長くても破綻しない
3000文字の質問で、先頭1000文字に切り詰める警告を出したうえで、回答と出典が出ることを確認。exit=0
- 再実行：同じ質問で複数回実行しても出典が欠けない
同じ質問を3回実行し、Sources が3回とも完全に一致することを diff で確認

## 根拠提示

- 出典の出し方：
検索でヒットしたチャンク（最大3件、スコア順）を、`- ファイルパス (excerpt: "抜粋")` の形で列挙する。抜粋はチャンク本文の先頭80文字を1行にしたもの。回答はこのチャンクだけを資料としてBedrockに渡して生成し、資料から答えられない場合（LLMが NO_ANSWER を返した場合）は、関係のない出典を出さずにヒットなしと同じ扱い（`- (none)`）にする

## リサーチメモ（任意）

調べたURLや、理解した要点をメモしてください。

- RAGの基本（分割、埋め込み、検索、プロンプトへの注入）
記事を段落単位で最大400文字のチャンクに分割。埋め込みは使わず、日本語の分かち書きが不要な「文字bigram（2文字ずつ）」の一致率で検索した（質問のbigramのうち何割がチャンクに含まれるか。ひらがなだけのbigramと「教えて」などの言い回しは除外）。チャンク単体では何の記事か分からず順位が落ちる問題があったため、検索時のみ記事タイトルをチャンクに付けて照合した。上位チャンクを <資料> タグで囲んでプロンプトに注入し、「資料だけを根拠に答える」よう指示（temperature=0）。Bedrock呼び出しは Day2 の invoke_bedrock を再利用 https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- 出典/ライセンスの確認方法
Wikipediaの文章は CC BY-SA 4.0。二次利用時は出典（記事名・URL）とライセンスの明記が必要 https://ja.wikipedia.org/wiki/Wikipedia:ウィキペディアを二次利用する 。本文の取得は MediaWiki API（action=query&prop=extracts&explaintext=1）を使用 https://www.mediawiki.org/wiki/Extension:TextExtracts
- 検索ヒットなし時の設計
①スコアが0.5未満のチャンクしかない場合、②ヒットはしたがLLMが資料から答えられないと判断した場合の2段階で「根拠なし」とする。関係の薄い資料を出典として出すと、根拠付きの回答だと誤解されるため。メッセージには対象データの一覧と次の行動（質問の言い換え／データの追加）を含め、処理自体は正常に終わっているため exit=0 とした
