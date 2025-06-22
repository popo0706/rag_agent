"""
【スクリプト概要】
LangChain 公式リポジトリから `.mdx` ドキュメントを取得し、
OpenAI の埋め込みモデルでベクトル化して Chroma に保存。
HyDE（Hypothetical Document Embeddings）テクニックを利用し、
ユーザの質問 → 仮想回答 → ベクトル検索 → GPT-4.1-nano で最終回答を生成する
簡易 RAG（Retriever-Augmented Generation）のサンプルです。

主な処理フロー
1. GitLoader でリポジトリをクローンし `.mdx` を抽出
2. OpenAIEmbeddings でドキュメントをベクトル化
3. Chroma に登録し retriever を生成
4. HyDE で仮想回答を作成し context を検索
5. プロンプトに質問 + context を挿入し GPT-4.1-nano が回答
"""

import os
from langchain_community.document_loaders import GitLoader  # GitHub リポジトリを扱う


# ---------- .mdx だけを読み込むフィルタ関数 ----------------------------------
def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")  # .mdx なら True


# ---------- リポジトリのクローン & ドキュメント読み込み -----------------------
repo_path = "./langchain"
if not os.path.exists(repo_path):
    # 初回実行時: リポジトリが無ければクローン
    loder = GitLoader(  # NOTE: 変数名は元コードに合わせて loder のまま
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter,
    )
    documents = loder.load()  # .mdx を Document オブジェクトに変換
else:
    # 既にクローン済み: そのまま読み込み
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter,
    )
    documents = loder.load()

print(len(documents))  # 何件読み込めたか確認

# ---------- ドキュメントをベクトル化して Chroma に登録 -----------------------
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 埋め込みモデル
db = Chroma.from_documents(documents, embeddings)  # Vector Store を作成

# ---------- LCEL（LangChain Expression Language）部品 ------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ▼ 質問と検索文脈を埋め込むプロンプトテンプレート ---------------------------
prompt = ChatPromptTemplate.from_template(
    """以下の文脈だけを参考に質問に答えてください。

文脈:
\"\"\"
{context}
\"\"\"

質問: {question}
"""
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 応答生成モデル
retriever = db.as_retriever()  # ベクトル検索インタフェース

# ---------- HyDE 用の仮想回答チェーン ----------------------------------------
hypotheical_prompt = ChatPromptTemplate.from_template(
    """次の質問に回答する一文を書いてください。

質問: {question}
"""
)

hypotheical_chain = hypotheical_prompt | model | StrOutputParser()  # 仮想回答を生成

# ---------- HyDE + RAG を組み合わせた最終チェーン ---------------------------
hyde_rag_chain = (
    {
        "question": RunnablePassthrough(),  # 質問をそのまま渡す
        "context": hypotheical_chain | retriever,  # 仮想回答を検索クエリに
    }
    | prompt  # プロンプトへ挿入
    | model  # LLM で回答
    | StrOutputParser()  # 文字列に整形
)

output = hyde_rag_chain.invoke("LangChainの概要を教えて")  # 動作確認
print(output)
