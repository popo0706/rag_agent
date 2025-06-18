# =============================================================================
# 【概要】
# このスクリプトは LangChain を使って「RAG（Retriever-Augmented Generation）」
# の超シンプル版を実装しています。流れは次の４ステップです。
#   1. Tavily の Web 検索 API で質問に関連する情報（文脈）を 3 件取得
#   2. ChatPromptTemplate に文脈と質問を差し込んでプロンプトを生成
#   3. GPT-4.1-nano モデルに推論させて回答を生成
#   4. StrOutputParser でモデルの出力を純粋な文字列に整形し表示
# 「東京の今日の天気は？」という質問を例に、検索結果を文脈として
# LLM が回答を返す一連の流れを学べます。
# コメントを付けているので、処理のつながりを確認しながら読んでみて
# ください。
# =============================================================================

from langchain_core.prompts import (
    ChatPromptTemplate,
)  # ❶ プロンプトのテンプレート（穴あき文章）を扱うクラス
from langchain_openai import (
    ChatOpenAI,
)  # ❷ OpenAI チャットモデルを呼び出すためのラッパー
from langchain_core.output_parsers import (
    StrOutputParser,
)  # ❸ モデルの出力を Python 文字列へ変換するパーサ

# ❹ ユーザーの質問と検索で得た文脈をはめ込むプロンプトを定義
prompt = ChatPromptTemplate.from_template(
    '''\
    以下の文脈だけを踏まえて質問に回答してください。
    
    文脈:"""
    {context}
    """
    
    質問:{question}
    '''
)

# ❺ gpt-4.1-nano を温度 0（決まりきった回答に寄せる設定）で初期化
model = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

from langchain_community.retrievers import (
    TavilySearchAPIRetriever,
)  # ❻ Tavily Web 検索用リトリーバをインポート

retriever = TavilySearchAPIRetriever(k=3)  # ❼ 上位 3 件の検索結果を取得するよう設定

from langchain_core.runnables import (
    RunnablePassthrough,
)  # ❽ 値をそのまま次ステップへ渡すユーティリティ

# ❾ 文脈生成 → プロンプト生成 → モデル推論 → 文字列変換 の４工程を
#    １本のチェーンとして定義（| 演算子で左から右にパイプ）
chain = (
    {
        "context": retriever,  # ① 検索結果を context キーにバインド
        "question": RunnablePassthrough(),
    }  # ② ユーザー質問をそのまま question にバインド
    | prompt  # ③ テンプレートに値を差し込みプロンプト完成
    | model  # ④ LLM で回答を生成
    | StrOutputParser()  # ⑤ モデル出力を文字列へ整形
)

# ⓭ 実際にチェーンを呼び出して回答を取得
output = chain.invoke("東京の今日の天気は？")
print(output)  # ⓮ コンソールに回答を表示
