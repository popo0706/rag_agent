"""
【概要】
LangChain と Pydantic を組み合わせて、ChatGPT 系 LLM から返ってくる JSON 文字列を
Pydantic モデル (Recipe) にそのままパースし、型安全に扱えるようにする最小サンプルです。

主な流れ:
 1) Recipe モデルを定義して食材 (ingredients) と手順 (steps) を厳密に型付け
 2) PydanticOutputParser にモデルを渡して専用パーサを作成
 3) ChatPromptTemplate で LLM に JSON 形式で出力するよう指示
 4) ChatOpenAI をラップし、response_format で JSON オブジェクトを要求
 5) prompt → model → output_parser をパイプ (|) で接続して実行
 6) 生成されたレシピを Recipe 型インスタンスとして安全に利用可能
"""

# -------------------------------------------------------------------
# LangChain 付属の「PydanticOutputParser」を読み込む
#   └ LLM が返す JSON テキストを自動で Pydantic モデルへ変換してくれる
from langchain_core.output_parsers import PydanticOutputParser  # <-- ① 解析器クラス

# -------------------------------------------------------------------
# Pydantic から BaseModel（基底クラス）と Field（フィールド定義用ヘルパ）
#   └ BaseModel : 型定義＋バリデーション機能を備えたクラス
#   └ Field     : 各フィールドに説明（description）や既定値などを付けられる
from pydantic import BaseModel, Field  # <-- ② データモデル定義用


# -------------------------------------------------------------------
# ■ Recipe モデル
#   AI が生成する「レシピ」を厳格な型で受け取るための定義
# -------------------------------------------------------------------
class Recipe(BaseModel):  # <-- ③ Pydantic モデルの宣言
    # 材料リスト（例: ["玉ねぎ", "にんじん", "鶏肉"]）
    # list[str] : 文字列を要素とするリストであることを示す型ヒント
    ingredients: list[str] = Field(
        description="ingredients of the dish"
    )  # <-- ④ Field に説明追加

    # 手順リスト（例: ["1. 野菜を切る", "2. 炒める"]）
    # description キーで「手順」を自然文で説明
    steps: list[str] = Field(
        description="steps to make the dish"
    )  # <-- ⑤ テンプレート内の説明


# -------------------------------------------------------------------
# Recipe モデルを渡して OutputParser をインスタンス化
#   └ LLM 出力を Recipe 型へ parse() してくれる
output_parser = PydanticOutputParser(pydantic_object=Recipe)  # <-- ⑥ モデル専用パーサ

from langchain_core.prompts import (
    ChatPromptTemplate,
)  # プロンプト文を組み立てるユーティリティ  # <-- ⑦

from langchain_openai import (
    ChatOpenAI,
)  # OpenAI LLM への LangChain ラッパ        # <-- ⑧

# ChatPromptTemplate.from_messages で system / human のメッセージ雛形を作成
prompt = ChatPromptTemplate.from_messages(  # <-- ⑨
    [
        # ── system メッセージ ──────────────────────────────
        #   AI へ「レシピを JSON で出力せよ」という役割指示を与える
        #   {format_instructions} は後で .partial() で具体的なフォーマット説明文を挿入
        (
            "system",
            "ユーザーが入力した料理のレシピを JSON 形式で出力してください。\n\n"
            "{format_instructions}",  # ← プレースホルダ                        # <-- ⑩
        ),
        # ── human メッセージ ───────────────────────────────
        #   実際の料理名（例: カレー）を {dish} に埋め込む
        ("human", "{dish}"),  # <-- ⑪
    ]
)

# prompt.partial(...) でプレースホルダを実際のフォーマット説明に置換
prompt_with_format_instructions = prompt.partial(  # <-- ⑫
    format_instructions=output_parser.get_format_instructions()
)

# ChatOpenAI インスタンスを生成し、出力を JSON オブジェクト形式に固定
model = ChatOpenAI(  # <-- ⑬
    model="gpt-4.1-nano",  # ← ユーザー指定モデル名
    temperature=0,  # ← 温度 0 で決定論的に
).bind(response_format={"type": "json_object"})

# `|` 演算子でチェーンを合成
#   prompt → model → output_parser の順に処理が流れる
chain = prompt_with_format_instructions | model | output_parser  # <-- ⑭

# 実行例: {"dish": "カレー"} を入力してレシピを生成
recipe = chain.invoke({"dish": "カレー"})  # <-- ⑮ LLM 呼び出し

# ターミナルに型と内容を出力（デバッグ用）
print(type(recipe))  # <-- ⑯ <class '__main__.Recipe'>
print(recipe)  # <-- ⑰ Recipe(ingredients=[...], steps=[...])
