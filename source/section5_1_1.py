# ===============================================================
# 【一言まとめ】
#   「料理名」を渡すと、そのレシピを生成 AI（gpt-4.1-nano）から
#   取得して表示するサンプルです。従来型（手続き型）と、
#   LangChain Expression Language（LCEL）で書いた
#   ２通りの実装方法を見比べられるようになっています。
# ---------------------------------------------------------------
# 【このスクリプトで学べること】
#   1. ChatPromptTemplate         …  システム発話＋ユーザ発話の定型化
#   2. ChatOpenAI                 …  OpenAI LLM（gpt-4.1-nano）の呼び出し
#   3. StrOutputParser            …  AIMessage から純テキストだけ抽出
#   4. LCEL（| 演算子）           …  Runnable をつないで宣言的に書く方法
# ===============================================================

# -------- ① 必要なライブラリをインポート --------
from langchain_core.output_parsers import StrOutputParser  # 文字列だけを取り出すパーサ
from langchain_core.prompts import ChatPromptTemplate  # プロンプト生成テンプレート
from langchain_openai import ChatOpenAI  # OpenAI LLM ラッパ

# -------- ② プロンプトを定義 --------
#   - "system" は AI への指示（必ずレシピを答えること）
#   - "human"  は ユーザ入力欄（{dish} が後で料理名に差し替わる）
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを教えてください。"),  # システム発話
        ("human", "{dish}"),  # ユーザ発話
    ]
)

# -------- ③ LLM（生成 AI）を用意 --------
#   model="gpt-4.1-nano" : 軽量モデル（応答が速く、コストも低め）
#   temperature=0        : ランダム性を抑えて毎回ほぼ同じ答えを返す
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

# -------- ④ 出力パーサを用意 --------
#   AIMessage → str へ変換。余計なメタ情報を削ぎ落として
#   「純粋なレシピ文字列」だけを得られるようにする
output_parser = StrOutputParser()

# ====================== ⑤ LCEL を使わない場合（従来型）======================

# Step-1: プレースホルダ {dish} に "カレー" を流し込み、
#         ChatPromptValue 型のデータを生成
prompt_value = prompt.invoke({"dish": "カレー"})

# Step-2: 生成した prompt_value を LLM にそのまま渡して推論
ai_message = model.invoke(prompt_value)

# Step-3: 返ってきた AIMessage からテキスト本文だけ抽出
output = output_parser.invoke(ai_message)

print(output)  # ← ここで「カレーのレシピ」が表示される

# ====================== ⑥ LCEL を使う場合（宣言的）======================

# Runnable 同士を “|” で直列につないでチェーンを組む
# （① プロンプト → ② LLM → ③ 出力パーサ）
chain = prompt | model | output_parser

# 使い方は簡単で、dish 名を渡すだけ。内部で①→②→③を自動実行
output2 = chain.invoke({"dish": "カレー"})
print(output2)  # ← 上と同じレシピが得られる

print("----------------------------------------------------------------------------")

# stream: 生成途中のテキストをストリームで受け取る例
for chunk in chain.stream({"dish": "カレー"}):
    print(chunk, end="", flush=True)

print()
print("----------------------------------------------------------------------------")

# batch: 複数リクエストをまとめて処理する例
output3 = chain.batch([{"dish": "カレー"}, {"dish": "うどん"}])
print(output3)  # ← ２品分のレシピがリストで返る
