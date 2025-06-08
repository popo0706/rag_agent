# ===============================================================
# 【要約】
# LangSmith（LangChain のプロンプト管理サービス）に登録済みの
# プロンプト「oshima/recipe」を呼び出し、変数 {dish} に
# 「カレー」を渡してレシピを生成・表示する最小サンプルです。
# 事前に環境変数で API エンドポイントやトレース設定を行い、
# LangSmith へ安全に接続できるようにしています。
# ===============================================================

import os  # os モジュール: 環境変数 (os.environ) を操作するために使用

# ──────────────────────────────────────────────
# LangSmith / LangChain に関する環境変数をコード内で直接セット
# ※ 実務では .env ファイル + load_dotenv() にまとめる方が安全
# ──────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 呼び出しログをクラウドへ送信
os.environ["LANGCHAIN_ENDPOINT"] = (
    "https://api.smith.langchain.com"  # LangSmith API エンドポイント
)
os.environ["LANGCHAIN_PROJECT"] = "agent-book"  # プロジェクト名（任意）

# ------------------------------------------------------------
# LangSmith で保存しておいた「プロンプト」を呼び出して実行する最小サンプル
# ------------------------------------------------------------

# 1) LangSmith の Python SDK から Client クラスをインポート
#    ── Client を使って登録済みプロンプトの取得・実行ができる
from langsmith import Client

# 2) Client のインスタンスを生成
#    - API キーやエンドポイント URL は環境変数
#      LANGCHAIN_API_KEY / LANGCHAIN_ENDPOINT などから読み込まれる
client = Client()

# 3) LangSmith に登録済みのプロンプトを取得
#    - "oshima/recipe" は「ユーザー名 / プロンプト名」の形式
prompt = client.pull_prompt("oshima/recipe")

# 4) 取り出したプロンプトを実行 (invoke)
#    - {dish} プレースホルダに "カレー" を渡し、レシピ生成を依頼
prompt_value = prompt.invoke({"dish": "カレー"})

# 5) 実行結果（生成されたレシピなど）を表示
#    - prompt_value は通常、文字列または ChatMessage 型
print(prompt_value)
