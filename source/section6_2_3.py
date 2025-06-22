"""
【概要】
GitHub 上の LangChain 公式リポジトリから `.mdx` ドキュメントを取得し、
OpenAI の埋め込みモデルでベクトル化したうえで Chroma に格納します。
さらに HyDE（仮想回答生成）テクニックで検索クエリを３通り作り、
関連ドキュメントを検索して GPT-4.1-nano に回答を生成させる
「簡易 RAG（Retriever-Augmented Generation）」のサンプルスクリプトです。
"""

import os
from langchain_community.document_loaders import (
    GitLoader,
)  # GitHub リポジトリ操作クラス

os.environ["LANGCHAIN_TRACING_V2"] = "true"  # 呼び出しログをクラウドへ送信
os.environ["LANGCHAIN_ENDPOINT"] = (
    "https://api.smith.langchain.com"  # LangSmith API エンドポイント
)
os.environ["LANGCHAIN_API_KEY"] = (
    "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"  # APIキー
)
os.environ["LANGCHAIN_PROJECT"] = "6-2-3"  # プロジェクト名（任意）


# ---------- .mdx だけを読み込むフィルタ関数 ----------
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")  # 拡張子が .mdx のとき True を返す


# ---------- リポジトリのクローンとドキュメント読み込み ----------
repo_path = "./langchain"  # クローン先フォルダ
if not os.path.exists(repo_path):  # フォルダ未作成＝初回実行
    loder = GitLoader(  # ※オリジナルの綴りどおり “loder” にしています
        clone_url="https://github.com/langchain-ai/langchain",  # リポジトリ URL
        repo_path=repo_path,  # クローン先
        branch="master",  # 取得ブランチ
        file_filter=file_filter,  # .mdx だけ読み込む
    )
    documents = loder.load()  # ドキュメントを読み込む
else:  # すでにクローン済み
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter,
    )
    documents = loder.load()  # ドキュメントを再読み込み

print(len(documents))  # 読み込めたドキュメント数を確認

# ---------- ドキュメントをベクトル化して Chroma に登録 ----------
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 埋め込みモデルを初期化
db = Chroma.from_documents(documents, embeddings)  # ベクトルストアを構築

# ---------- LCEL（LangChain Expression Language）部品 ----------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 質問と検索文脈を埋め込むプロンプトテンプレート
prompt = ChatPromptTemplate.from_template(
    """以下の文脈だけを参考に質問に答えてください。

文脈:
\"\"\"
{context}
\"\"\"

質問: {question}
"""
)

model = ChatOpenAI(
    model="gpt-4.1-nano", temperature=0
)  # 生成モデル（温度 0 で決定論的）
retriever = db.as_retriever()  # ベクトル検索インタフェースを取得

from pydantic import BaseModel, Field


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(
        ..., description="検索クエリのリスト"
    )  # Pydantic で構造化出力を定義


# HyDE 用：質問から検索クエリを 3 個生成するプロンプト
query_generation_prompt = ChatPromptTemplate.from_template(
    """
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似検索の限界を克服するために、
ユーザの質問に対して複数の視点を提供することが目標です。

質問：{question}
"""
)

# 質問 → クエリ生成 → リスト抽出 のチェーン
query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(
        QueryGenerationOutput
    )  # 構造化出力でクエリリストを取得
    | (lambda x: x.queries)  # queries フィールドだけ取り出す
)

# Multi-Query RAG：複数クエリで検索し文脈をまとめて回答を生成
multi_query_rag_chain = (
    {
        "question": RunnablePassthrough(),  # ユーザ質問をそのまま渡す
        "context": query_generation_chain
        | retriever.map(),  # 生成した各クエリで検索した結果を結合
    }
    | prompt  # プロンプトに埋め込む
    | model  # LLM で回答生成
    | StrOutputParser()  # 出力を文字列へ変換
)

output = multi_query_rag_chain.invoke("LangChain の概要を教えて")  # 実行例
print(output)  # 生成された回答を表示
