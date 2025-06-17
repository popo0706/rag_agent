# =============================================================================
# 【概要】
# このスクリプトは、ユーザが入力した「topic（話題）」について
#   1) 楽観主義者としての意見
#   2) 悲観主義者としての意見
# をそれぞれ GPT-4.1-nano に生成させ、最後に両方を辞書形式で表示する
# デモンストレーションです。
#
# 処理の大まかな流れ
# ---------------------------------------------------------------------------
# ①  ライブラリのインポート
# ②  LLM（GPT-4.1-nano）と出力パーサの初期化
# ③  “楽観的” プロンプトとチェーンの定義
# ④  “悲観的” プロンプトとチェーンの定義
# ⑤  2 本のチェーンを RunnableParallel で並列実行するセットを作成
# ⑥  ユーザ入力（topic）を渡して両チェーンを同時に実行し、結果を表示
# =============================================================================

# ① ライブラリのインポート ---------------------------------------------------
from langchain_core.output_parsers import StrOutputParser  # LLM の出力を文字列へ
from langchain_core.prompts import ChatPromptTemplate  # system/human プロンプト用
from langchain_openai import ChatOpenAI  # OpenAI チャットモデル

# ② LLM と出力パーサの初期化 -------------------------------------------------
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 生成モデルを固定・温度 0
output_parser = StrOutputParser()  # 文字列へ変換

# ③ “楽観的” プロンプトとチェーン --------------------------------------------
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

# ④ “悲観的” プロンプトとチェーン --------------------------------------------
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

# ⑤ 2 本のチェーンを並列実行する設定 ----------------------------------------
import pprint  # 出力を見やすく整形
from langchain_core.runnables import RunnableParallel  # 並列実行ユーティリティ

paralell_chain = RunnableParallel(  # 2 本のチェーンをまとめる
    {
        "optimistic_option": optimistic_chain,  # 楽観的な回答
        "pessimistic_option": pessimistic_chain,  # 悲観的な回答
    }
)

# ⑥ 実行と結果表示 -----------------------------------------------------------
output = paralell_chain.invoke({"topic": "生成AIの進化について"})  # topic を渡して実行
pprint.pprint(output)  # 辞書形式で見やすく出力
