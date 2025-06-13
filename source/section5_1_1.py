# ===============================================================
# 【概要】
# このスクリプトは「料理名」を入力すると、そのレシピを生成 AI
# （OpenAI の gpt-4.1-nano モデル）に問い合わせて取得するサンプルです。
# ───────────────────────────────────────────
# 1) ChatPromptTemplate で “システム発話＋ユーザ発話” の形に整形
# 2) ChatOpenAI で LLM を呼び出し、AIMessage を受け取る
# 3) StrOutputParser で AIMessage から “純粋な文字列” を抽出
# 4) LCEL を「使わない方法」と「使う方法」の 2 パターンを比較
#    - LCEL = LangChain Expression Language
#      “|” 演算子で Runnable をつなげ、宣言的にワークフローを構築できる
# ===============================================================

from langchain_core.output_parsers import StrOutputParser  # ← ③ 文字列抽出パーサ
from langchain_core.prompts import ChatPromptTemplate  # ← ① プロンプト生成テンプレ
from langchain_openai import ChatOpenAI  # ← ② OpenAI LLM ラッパ

# ---------------------------------------------------------------
# ■ プロンプトを定義
#    - system: モデルへの指示（レシピを答えてね）
#    - human : ユーザが入力する料理名（dish フィールドに後で差し込む）
# ---------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーが入力した料理のレシピを教えてください。",
        ),  # システムメッセージ
        ("human", "{dish}"),  # ユーザ入力（プレースホルダ）
    ]
)

# ---------------------------------------------------------------
# ■ LLM を準備
#    - model        : gpt-4.1-nano（軽量モデルを想定）
#    - temperature  : 0 なので“決まった答え”を返しやすい（ランダム性を抑制）
# ---------------------------------------------------------------
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# ---------------------------------------------------------------
# ■ 出力パーサ
#    - AIMessage → str へ変換してくれるユーティリティ
# ---------------------------------------------------------------
output_parser = StrOutputParser()

# ================================ LCEL を使わない場合 ================================

# ---- Step-1: prompt.invoke でテンプレートに値を流し込み、ChatPromptValue を取得
prompt_value = prompt.invoke({"dish": "カレー"})
# ---- Step-2: LLM を直接呼び出し、AIMessage を受け取る
ai_message = model.invoke(prompt_value)
# ---- Step-3: AIMessage.content だけを取り出す
output = output_parser.invoke(ai_message)

print(output)  # ⇒ カレーのレシピが表示される

# ================================ LCEL を使う場合 ================================

# “|” で Runnable を直列につなげ、1 行でチェーン化
chain = prompt | model | output_parser

# 同じく dish="カレー" を渡すだけで、内部で ①→②→③ が実行される
output2 = chain.invoke({"dish": "カレー"})

print(output2)  # ⇒ 上と同じレシピが得られる（処理の書き方が宣言的になる）
