# =============================================================================
# 【スクリプト全体の要約】
# このプログラムは――
#   1. GitHub の LangChain 公式リポジトリをクローンし、拡張子 .mdx の
#      ドキュメントだけを読み込む（初回だけクローン、2回目以降は再利用）
#   2. OpenAI の埋め込みモデルでドキュメントをベクトル（数字の並び）に変換し
#      Chroma というデータベースに保存する
#   3. 質問文から HyDE 手法で 3 つの検索クエリを自動生成し、
#      それぞれでベクトル検索して文脈（context）を集める
#   4. 集めた文脈をプロンプトに差し込み、GPT-4.1-nano に答えを作らせる
#   5. 追加ステップとして Reciprocal Rank Fusion（複数検索結果の再順位付け）
#      を行い、より良い文脈を使って再度回答を生成する
# LangChain・RAG・HyDE という 3 つのテクニックをまとめて学べる
# 教材向けのサンプルコードです。
# =============================================================================

"""
【概要】
GitHub 上の LangChain 公式リポジトリから `.mdx` ドキュメントを取得し、
OpenAI の埋め込みモデルでベクトル化したうえで Chroma に格納します。
さらに HyDE（仮想回答生成）テクニックで検索クエリを３通り作り、
関連ドキュメントを検索して GPT-4.1-nano に回答を生成させる
「簡易 RAG（Retriever-Augmented Generation）」のサンプルスクリプトです。
"""

# ---------- 必要なライブラリのインポート ----------
import os  # OS の環境変数操作に使う標準ライブラリ
from langchain_community.document_loaders import (
    GitLoader,  # GitHub リポジトリを扱う LangChain クラス
)

# ---------- LangSmith（LangChain のログ可視化サービス）の設定 ----------
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # トレーシングを有効化
project_name = os.path.splitext(os.path.basename(__file__))[0]
os.environ["LANGCHAIN_PROJECT"] = (
    project_name  # プロジェクト名（ファイル名から自動設定）
)


# ---------- .mdx ファイルだけを通すフィルタ関数 ----------
def file_filter(file_path: str) -> bool:  # 引数はファイルのパス文字列
    return file_path.endswith(".mdx")  # 拡張子が .mdx なら True


# ---------- リポジトリのクローンとドキュメント読み込み ----------
repo_path = "./langchain"  # クローン先のフォルダ名
if not os.path.exists(repo_path):  # フォルダが無い＝初回実行
    loder = GitLoader(  # ※綴りが「loder」なのは原文そのまま
        clone_url="https://github.com/langchain-ai/langchain",  # 公式リポジトリ
        repo_path=repo_path,  # 保存先
        branch="master",  # 取得ブランチ
        file_filter=file_filter,  # .mdx だけ読み込むためのフィルタ
    )
    documents = loder.load()  # クローンしてドキュメント生成
else:  # 既にクローン済みなら再利用
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter,
    )
    documents = loder.load()  # 最新状態に更新して読み込み

print(len(documents))  # 読み込めたドキュメント数を確認

# ---------- ドキュメントをベクトル化して Chroma に登録 ----------
from langchain_chroma import Chroma  # ベクトルストア実装
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # LLM と埋め込み

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 埋め込みモデル
db = Chroma.from_documents(documents, embeddings)  # ベクトルストア生成
retriever = db.as_retriever()  # 検索インタフェース取得

# ---------- プロンプトとパーサーなど LCEL 部品 ----------
from langchain_core.output_parsers import StrOutputParser  # 出力を文字列に変換
from langchain_core.prompts import ChatPromptTemplate  # system/human テンプレ
from langchain_core.runnables import RunnablePassthrough  # 値を素通しするノード

# 文脈と質問を差し込むテンプレート
prompt = ChatPromptTemplate.from_template(
    """以下の文脈だけを参考に質問に答えてください。

文脈:
\"\"\"
{context}
\"\"\"

質問: {question}
"""
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 決定論的な LLM


from typing import Any

from langchain_cohere import CohereRerank
from langchain_core.documents import Document


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_reranker.compress_documents(documents=documents, query=question)


rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank)
    | prompt
    | model
    | StrOutputParser()
)

output = rerank_rag_chain.invoke("Langchainの概要を教えて")
print(output)
