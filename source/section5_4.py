# =============================================================================
# 【概要】
# このスクリプトは LangChain を用いて、ユーザが入力した「topic」について
#   1. 楽観主義者（optimistic）と悲観主義者（pessimistic）の 2 種類の立場で
#      それぞれ意見を生成するチェーンを作成
#   2. RunnableParallel で 2 本のチェーンを“同時に”実行し高速に結果を取得
#   3. 得られた 2 つの意見を客観的 AI が統合し、バランスの取れた結論を導出
# する処理の流れを実装しています。
# 実行結果は pprint で見やすく表示されます。
# =============================================================================

from langchain_core.output_parsers import StrOutputParser  # LLM の出力を文字列へ
from langchain_core.prompts import ChatPromptTemplate  # system/human プロンプト用
from langchain_openai import ChatOpenAI  # OpenAI チャットモデル

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 生成モデルを固定・温度 0
output_parser = StrOutputParser()  # 文字列へ変換

# ---------- 楽観主義者チェーン ------------------------------------------------
optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは楽観主義者です。ユーザの入力に対して楽観的な意見をください。",
        ),
        ("human", "{topic}"),  # {topic} にユーザ入力が入る
    ]
)
optimistic_chain = (
    optimistic_prompt | model | output_parser
)  # プロンプト → モデル → 変換

# ---------- 悲観主義者チェーン ------------------------------------------------
pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは悲観主義者です。ユーザの入力に対して悲観的な意見をください。",
        ),
        ("human", "{topic}"),
    ]
)
pessimistic_chain = (
    pessimistic_prompt | model | output_parser
)  # プロンプト → モデル → 変換

import pprint  # 出力を見やすく整形
from langchain_core.runnables import RunnableParallel  # 並列実行ユーティリティ

# ---------- 結論統合チェーン ---------------------------------------------------
synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
        (
            "human",
            "楽観的意見：{optimistic_option}\n悲観的意見：{pessimistic_option}",
        ),
    ]
)

synthesize_chain = (
    RunnableParallel(  # 2 本のチェーンを並列でまとめる
        {
            "optimistic_option": optimistic_chain,  # 楽観的な回答
            "pessimistic_option": pessimistic_chain,  # 悲観的な回答
        }
    )
    | synthesize_prompt  # 2 つの意見を統合するプロンプト
    | model  # 客観的 AI として推論
    | output_parser  # 文字列へ変換
)

# ---------- 実行 --------------------------------------------------------------
output = synthesize_chain.invoke(
    {"topic": "生成AIの進化について"}  # topic を渡して実行
)
pprint.pprint(output)  # 辞書形式で見やすく出力
