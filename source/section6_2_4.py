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

# ---------- Pydantic で検索クエリの構造を定義 ----------
from pydantic import BaseModel, Field


class QueryGenerationOutput(BaseModel):
    # 「queries」という名前のフィールドを定義
    # 型は list[str] ＝ 文字列のリスト
    # Field(..., description="検索クエリのリスト")
    #   - ...   : 「必須」を意味する予約語（値が空だとエラー）
    #   - description : ドキュメント用の説明文
    queries: list[str] = Field(..., description="検索クエリのリスト")


# ---------- 質問 → 検索クエリ（3つ）を生成するプロンプト ----------
query_generation_prompt = ChatPromptTemplate.from_template(
    """
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似検索の限界を克服するため、
ユーザの質問に対して複数の視点を提供することが目標です。

質問：{question}
"""
)

# ---------- 質問 → クエリ生成 → クエリリスト抽出 のチェーン ----------
query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)  # 構造化出力で受け取る
    | (lambda x: x.queries)  # queries フィールドのみ残す
)

# ---------- Multi-Query RAG（HyDE クエリを全部使う） ----------
multi_query_rag_chain = (
    {
        "question": RunnablePassthrough(),  # ユーザ質問をそのまま渡す
        "context": query_generation_chain
        | retriever.map(),  # 各クエリを使って検索結果を合体
    }
    | prompt  # プロンプトに組み込む
    | model  # LLM で回答生成
    | StrOutputParser()  # 出力を純粋な文字列へ
)

output = multi_query_rag_chain.invoke("LangChain の概要を教えて")  # 実行例
print(output)  # 生成された回答を表示
print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("--------------------------------------------------------")

# ---------- Reciprocal Rank Fusion（検索結果の再順位付け） ----------
from langchain_core.documents import Document  # 取得文書の型


def recipocal_rank_fusion(
    retriever_outputs: list[Document],  # 検索結果のリスト
    k: int = 60,  # 重み付け用の定数
) -> list[str]:
    """
    検索結果を「上位に来た回数」と「順位」に応じて再スコアリングする
    かんたんな実装例。ここでは本文（page_content）だけをキーにしている。
    """
    content_score_mapping = {}  # 文書内容とスコアを紐づける辞書

    for docs in retriever_outputs:  # 各クエリごとの結果に対して
        for rank, doc in enumerate(docs):  # 順位と文書を取り出す
            content = doc.page_content  # 実際のテキスト
            if content not in content_score_mapping:
                content_score_mapping[content] = 0
            content_score_mapping[content] += 1 / (rank + k)  # RRF の公式

    ranked = sorted(  # スコアが高い順に並べ替え
        content_score_mapping.items(), key=lambda x: x[1], reverse=True
    )
    return [content for content, _ in ranked]  # 文書本文だけをリストで返す


rag_fusion_chain = (
    {
        "question": RunnablePassthrough(),  # 質問文
        "context": query_generation_chain | retriever.map() | recipocal_rank_fusion,
    }
    | prompt
    | model
    | StrOutputParser()  # ※ () を付けて呼び出しオブジェクト化
)

output = rag_fusion_chain.invoke("LangChainの概要を教えて")  # 再順位付け版の実行
print(output)
