# ===============================================================
# 【要約】
#  - Recipe という Pydantic モデルを定義し、
#  - output_parser.get_format_instructions() で
#    “この JSON 形式で返してね” という指示文を自動生成。
#  - その指示文を ChatPromptTemplate に差し込み、
#  - {dish} だけ可変にして「カレーのレシピ」を取得する。
#
# ＊＊＊「format_instructions をどう渡すか？」の解説＊＊＊
# 1) 毎回同じ値なら `prompt.partial(format_instructions=...)` で
#    先に固定すると、invoke 時に渡すのは {dish} だけで済む。
# 2) 一度しか呼ばない／動的に変えるなら、partial を使わず
#    `prompt.invoke({ "dish": ..., "format_instructions": ... })`
#    とまとめて渡しても動作は変わらない。
# ここでは “再利用を想定して固定値だけ前もってバインドする”
# という教科書的パターンを見せるために partial を採用。
# ===============================================================


# ==============================================================
# 1) 必要なライブラリの読み込み
# ==============================================================

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ==============================================================
# 2) レシピ情報を表す Pydantic モデル
# ==============================================================


class Recipe(BaseModel):
    """AI が出力するレシピを受け取るための構造定義。"""

    ingredients: list[str] = Field(
        description="料理の材料をリスト形式で列挙してください"
    )
    steps: list[str] = Field(description="1 手順 1 行で書いた調理の工程")


# ==============================================================
# 3) OutputParser を用意し、フォーマット指示文を取得
# ==============================================================

output_parser = PydanticOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()

# （デバッグ用に確認したい人は↓を有効化してください）
# print("▼format_instructions\n", format_instructions, "\n", "-"*60)


# ==============================================================
# 4) ChatPromptTemplate を定義
# ==============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザが入力した料理のレシピを教えてください。\n\n"
            "{format_instructions}",  # ← ここに指示文を埋め込む
        ),
        ("human", "{dish}"),  # ← ユーザが指定する料理名
    ]
)

# ---------------- ここが今回のポイント ----------------
#  ■ format_instructions を事前に固定（partial）
#     - 再利用時に {dish} だけ渡せば良い
#  ■ もちろん次のように一括でも OK
#     prompt_value = prompt.invoke({"dish": "カレー",
#                                   "format_instructions": format_instructions})
#  ------------------------------------------------------
prompt_fixed = prompt.partial(format_instructions=format_instructions)


# ==============================================================
# 5) 実際にテンプレートを展開（ここでは “カレー”）
# ==============================================================

prompt_value = prompt_fixed.invoke({"dish": "カレー"})

print("=== role:system =========================================")
print(
    prompt_value.messages[0].content
)  # システムメッセージ（指示＋JSONフォーマット指定）
print("=== role:user ===========================================")
print(prompt_value.messages[1].content)  # ユーザーメッセージ（今回の料理名）
