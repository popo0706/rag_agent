# =============================================================================
# 【概要】
# このスクリプトは LangChain の公式リポジトリから .mdx ドキュメントを取得し、
# OpenAI の埋め込みモデルでベクトル化して Chroma ベクトルストアに保存。
# その後、ユーザの質問内容をキーにドキュメント検索（Retriever）を行い、
# 取得した文脈をプロンプトに差し込んで GPT-4.1-nano で回答を生成する
# 「簡易 RAG（Retrieval-Augmented Generation）」の実装例です。
# - ステップ1 : GitLoader でリポジトリをクローン＆ドキュメント抽出
# - ステップ2 : OpenAIEmbeddings でドキュメントを数値ベクトル化
# - ステップ3 : Chroma へ登録し retriever を取得
# - ステップ4 : ユーザ質問＋検索結果をプロンプトに渡して LLM で回答生成
# =============================================================================

import os
from langchain_community.document_loaders import GitLoader  # Git リポ読み込みに使用


# --- 対象ファイルを .mdx に限定するフィルタ関数 -------------------------------
def file_filter(file_path: str) -> bool:  # GitLoader から呼ばれる
    return file_path.endswith(".mdx")  # True なら読み込み対象


# --- Git リポジトリをクローンしてドキュメントを取得 ---------------------------
repo_path = "./langchain"
if not os.path.exists(repo_path):
    # リポジトリが存在しない場合のみクローンする
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",  # 公式 LangChain リポ
        repo_path=repo_path,  # ローカル保存先
        branch="master",  # 取得ブランチ
        file_filter=file_filter,  # 上記フィルタ適用
    )
    documents = loder.load()  # .mdx を Document 化
else:
    # 既にリポジトリが存在する場合はそのまま読み込む
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",  # 公式 LangChain リポ
        repo_path=repo_path,  # ローカル保存先
        branch="master",  # 取得ブランチ
        file_filter=file_filter,  # 上記フィルタ適用
    )
    documents = loder.load()  # .mdx を Document 化

print(len(documents))  # 取得件数を確認

# --- ドキュメントをベクトル化し Chroma に登録 -------------------------------
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 埋め込み用モデル
db = Chroma.from_documents(documents, embeddings)  # Vector Store 構築

# --- LCEL（LangChain Expression Language）の部品 ----------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# プロンプトテンプレート：検索結果（context）と質問（question）を差し込む
prompt = ChatPromptTemplate.from_template(
    '''\
以下の文脈だけ踏まえて質問に回答してください。

文脈："""
{context}
"""

質問：{question}
'''
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 応答生成モデル

retriever = db.as_retriever()  # ベクトル検索インタフェース

# --- 質問→検索→生成をひとつのチェーンに結合 -------------------------------
chain = (
    {
        "question": RunnablePassthrough(),  # 質問文字列をそのまま渡す
        "context": retriever,  # 検索結果（文脈）を渡す
    }
    | prompt  # プロンプトを組み立て
    | model  # LLM で回答生成
    | StrOutputParser()  # 出力を純粋な文字列に変換
)

# --- チェーン実行例 ----------------------------------------------------------
output = chain.invoke("Langchainの概要を教えて")  # 結果は戻り値で取得

print(output)
