# =============================================================================
# 【要約】
# 本スクリプトは LangChain を用いて “RAG（Retriever-Augmented Generation）”
# を最小構成で実装したものです。
# 手順は次の４段階です。
#   1. TavilySearchAPIRetriever でユーザーの質問に関連する文脈（上位３件）を取得
#   2. ChatPromptTemplate に文脈と質問を差し込み、プロンプトを作成
#   3. GPT-4.1-nano に推論させ、回答を生成
#   4. StrOutputParser で回答を文字列化し、質問・文脈・回答を辞書形式で出力
# “東京の今日の天気は？”という質問を例に、実際の検索結果を文脈として
# LLM が回答を返す流れを学べます。
# =============================================================================

# --- 必要なクラス／関数をインポート ------------------------------------------------
from langchain_core.prompts import (
    ChatPromptTemplate,  # プロンプトテンプレート（穴あき文章）を扱う
)
from langchain_openai import (
    ChatOpenAI,  # OpenAI チャットモデルのラッパー
)
from langchain_core.output_parsers import (
    StrOutputParser,  # LLM 出力を Python 文字列へ変換
)

# --- 質問と検索文脈を差し込むプロンプトテンプレートを定義 --------------------------
prompt = ChatPromptTemplate.from_template(
    '''\
    以下の文脈だけを踏まえて質問に回答してください。

    文脈:"""
    {context}
    """

    質問:{question}
    '''
)

# --- LLM（GPT-4.1-nano）を初期化 ----------------------------------------------------
model = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)  # 温度0で決定論寄り

# --- Web 検索で文脈を取得するリトリーバを用意 --------------------------------------
from langchain_community.retrievers import (
    TavilySearchAPIRetriever,  # Tavily Web 検索 API 用リトリーバ
)

retriever = TavilySearchAPIRetriever(k=3)  # 上位３件を文脈として取得する設定

# --- Runnable ユーティリティのインポート ------------------------------------------
from langchain_core.runnables import (
    RunnablePassthrough,  # 値をそのまま次ステップへ渡すだけのラッパー
)

import pprint  # 結果を見やすく整形して表示するための標準ライブラリ

# --- チェーン（処理パイプライン）を組み立てる ------------------------------------
"""
① “| prompt | model | StrOutputParser()” で終わる形
→ 最後の部品が 文字列を返す ので、チェーン全体も「回答の文字列」だけを返す。

② 最後に RunnablePassthrough.assign(...) を挟む形
→ assign が 「元の辞書」＋「answer という新しいキー」 を返すため、質問・文脈・回答すべてが入った辞書になる。
"""
chain = {
    "question": RunnablePassthrough(),  # ユーザー入力を “question” にそのまま渡す
    "context": retriever,  # ユーザー入力を渡して文脈（検索結果）を取得
} | RunnablePassthrough.assign(  # 取得した dict に “answer” キーを追加
    answer=prompt | model | StrOutputParser()  # プロンプト → LLM → 文字列化
)


# --- チェーンを実行して結果を表示 --------------------------------------------------
output = chain.invoke("東京の今日の天気は？")  # 質問を投入してパイプライン実行
pprint.pprint(output)  # 辞書形式の結果を整形出力
