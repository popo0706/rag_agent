# ===============================================================
# 【一言まとめ】
#   Zero-shot CoT（Chain-of-Thought）で「ステップごとの思考＋
#   結論抽出」を 1 本のチェーンにまとめるサンプルです。
#   ───────────────────────────────
#   ① ChatPromptTemplate で “質問 → ステップバイステップ回答”
#      のプロンプトを作成
#   ② LLM（gpt-4.1-nano）で推論し、思考過程を含む回答を取得
#   ③ その回答から「結論だけを抽出する」プロンプトを作り直し
#   ④ ①→②→③ をパイプ（|）で連結し、最終的な答えを得る
# ★冗長な思考過程を省き、結論をシンプルに出力（1度のLLMの呼び出しで難しい書き方にしなくてよい効果もある。）
# ===============================================================

from langchain_core.output_parsers import StrOutputParser  # ← 文字列抽出パーサ
from langchain_core.prompts import ChatPromptTemplate  # ← プロンプト生成テンプレ
from langchain_openai import ChatOpenAI  # ← OpenAI LLM ラッパ

# ---------------------------------------------------------------
# ■ 生成 AI（LLM）の準備
#    - gpt-4.1-nano : 軽量モデルで応答が速い
#    - temperature  : 0 → 出力のランダム性を抑え、安定した答えにする
# ---------------------------------------------------------------
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# ---------------------------------------------------------------
# ■ 出力パーサ
#    - AIMessage → str へ変換
# ---------------------------------------------------------------
output_parser = StrOutputParser()

# ====================== ① Zero-shot CoT 用プロンプト ======================

# 「ステップバイステップで考えてください」と明示することで
# モデルに思考過程を出力させる
cot_promt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),  # ← 後で {question} を実際の質問で置換
    ]
)

# Runnable チェーン: プロンプト → LLM → str 抽出
cot_chain = cot_promt | model | output_parser

# ====================== ② 結論抽出用プロンプト ======================

# 先ほどの“思考過程込み回答”を {text} として渡し、
# 結論だけを抜き出してもらう
summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけを抽出してください。"),
        ("human", "{text}"),
    ]
)
summarize_chain = summarize_prompt | model | output_parser

# ====================== ③ 二段構えの最終チェーン ======================

# Step-1: Zero-shot CoT で思考過程つき回答を得る
# Step-2: その回答を要約プロンプトに渡して結論だけ返す
cot_summarize_chain = cot_chain | summarize_chain

# ====================== ④ 実行例 ======================

output = cot_summarize_chain.invoke({"question": "10+2*3"})
print(output)  # ⇒ 16
