# =============================================================================
# 【概要】
# このスクリプトは LangChain を使い、1 つの「トピック（topic）」に対して
#   ① 楽観的な意見
#   ② 悲観的な意見
#   ③ ①②を統合した客観的な結論
# ――という 3 つの視点を連続して生成・出力するデモです。
# それぞれ独立したチェーン（optimistic_chain / pessimistic_chain）を作り、
# 最後に synthesize_chain でまとめています。
# =============================================================================

# ----------------------------- インポート -------------------------------------
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser  # LLM の出力を文字列へ変換
from langchain_core.prompts import ChatPromptTemplate  # system/human 形式プロンプト
from langchain_openai import ChatOpenAI  # OpenAI チャットモデル

# ----------------------------- 共通設定 ---------------------------------------
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 生成モデルを固定・温度 0
output_parser = StrOutputParser()  # モデル出力 → 文字列へ

# ----------------------- 楽観主義者チェーン -----------------------------------
# 「topic」を受け取り、楽観的な意見を生成する
optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたは楽観主義者です。ユーザの入力に対して楽観的な意見をください。",
        ),
        ("human", "{topic}"),  # {topic} プレースホルダにユーザ入力が入る
    ]
)
optimistic_chain = (
    optimistic_prompt | model | output_parser  # プロンプト → モデル → 文字列
)

# ----------------------- 悲観主義者チェーン -----------------------------------
# 「topic」を受け取り、悲観的な意見を生成する
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
    pessimistic_prompt | model | output_parser  # プロンプト → モデル → 文字列
)

# ----------------------- 統合（客観的）チェーン -------------------------------
# 上記 2 つの意見を受け取り、客観的に要約・統合する
synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
        ("human", "楽観的意見：{optimistic_option}\n悲観的意見：{pessimistic_option}"),
    ]
)

systhesize_chain = (  # ❶ 楽観・悲観チェーンを並列実行し、②で統合
    {
        "optimistic_option": optimistic_chain,  # ① 楽観的意見
        "pessimistic_option": pessimistic_chain,  # ① 悲観的意見
        "topic": itemgetter("topic"),  # 呼び出し時に渡された "topic" をそのまま流用
    }
    | synthesize_prompt  # ② 2 つの意見をまとめるプロンプト
    | model  # ③ LLM で統合
    | output_parser  # ④ 文字列化
)

# ----------------------------- 実行 -------------------------------------------
output = systhesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)  # 統合された客観的な意見を表示
