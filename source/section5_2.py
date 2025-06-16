# =============================================================
# 【超概要：このスクリプトでやること】
# LangChain を使って
#   ① プロンプト作成 → ② GPT-4.1-nano で推論 → ③ 出力を文字列へ変換
#   ④ さらに大文字／小文字へ整形
# ――という 4 段階を 1 本のチェーンで実行し、
# 例として "Hello" → "HELLO"／"HELLOW" → "hellow" を確認するデモです。
# =============================================================

# =============================================================================
# 【概要】
# このスクリプトは LangChain を使って、次の 4 段階を 1 本のチェーンで実行します。
#   1. 「system」「human」メッセージでプロンプトを組み立てる
#   2. GPT-4.1-nano に推論させる
#   3. 返ってきたメッセージを純粋な文字列へ変換する
#   4. その文字列をすべて大文字（UPPER CASE）に変換する
# 入力に "Hello" を与えると、最終出力は "HELLO" になります。
# =============================================================================

# --- 文字列抽出用のパーサーを取り込む ----------------------------
from langchain_core.output_parsers import StrOutputParser  # ❶ LLM 出力を文字列へ変換

# --- プロンプトテンプレートを取り込む -----------------------------
from langchain_core.prompts import (
    ChatPromptTemplate,
)  # ❷ system/human 形式プロンプトのテンプレート

# --- OpenAI チャットモデルを扱うラッパー ---------------------------
from langchain_openai import ChatOpenAI  # ❸ OpenAI チャットモデル用ラッパー

# ❹ プロンプトを定義: system メッセージ＋ユーザー入力プレースホルダ
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),  # ← モデルの役割指示
        ("human", "{input}"),  # ユーザー入力が {input} に入る
    ]
)

# --- GPT-4.1-nano を決定論的に呼び出す -----------------------------
model = ChatOpenAI(
    model="gpt-4.1-nano", temperature=0
)  # ❺ 決定論的 (temperature=0) に応答

# --- LLM から返る Message → str へ変換するパーサー ---------------
output_parser = StrOutputParser()  # ❻ Message → str へ変換

# --- 任意の Python 関数をチェーンに組み込むユーティリティ ----------
from langchain_core.runnables import (
    RunnableLambda,
)  # ❼ 関数をチェーンに組み込むユーティリティ


# --- 文字列を大文字に変換するユーティリティ関数 --------------------
def upper(text: str) -> str:  # ❽ 文字列を大文字に変換する関数
    return text.upper()


# ❾ プロンプト → モデル → 文字列抽出 → 大文字化 を「|」で接続
chchain_prompt = prompt | model | output_parser | RunnableLambda(upper)

# prompt | model |  RunnableLambda(upper)　だとエラーになる。modelがAIメッセージを返し、RunnableLambda(upper)はstrが必要なため！

# chain_prompt = prompt | model | output_parser | upper #|で連結されて片方がRunnableLambdaで、もう片方が関数であれば自動的ににRunnableLambdaに変換される。

# ❿ チェーン実行: {input: "Hello"} を渡す
output = chchain_prompt.invoke({"input": "Hello"})
print(output)  # ⓫ → "HELLO"

# --- @chain デコレータを使ったチェーン化の別パターン --------------
from langchain_core.runnables import chain


@chain  # LangChain のデコレータで「関数＝チェーン」として扱う
def lower(text: str) -> str:
    return text.lower()


# --- lower() を末尾に挟んで小文字化バージョンを作る ---------------
chain_prompt2 = prompt | model | output_parser | lower

# --- 実行例  --------------------------------------------------------
output2 = chain_prompt2.invoke({"input": "HELLOW"})
print(output2)  # → "hellow"
