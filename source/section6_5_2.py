# =============================================================================
# 【概要】
# このスクリプトは「LangChain 公式リポジトリの .mdx ドキュメントを取得して
# ベクトル検索を行い、質問に応じて LangChain 文書 or Web 検索を自動で
# 切り替えて回答を生成する」一連の処理を示しています。
# 主な処理の流れ
#   1. 環境変数を設定し LangSmith のトレーシングを有効化
#   2. GitLoader で公式リポジトリをクローンし .mdx を読み込み
#   3. OpenAIEmbeddings でベクトル化し Chroma に登録
#   4. Retriever を 2 系統（LangChain 文書 / Web）用意
#   5. 質問内容に応じて最適な Retriever を自動選択
#   6. 取得した文脈をプロンプトに埋め込み GPT-4.1-nano で回答生成
# 新人向けポイント
#   - 「環境変数」「埋め込みモデル」「Retriever」「プロンプト」の
#     役割を意識すると理解しやすい
# =============================================================================

# ---------- 1. 必要な標準ライブラリのインポート ----------
import os  # OS に依存する処理（パス操作や環境変数操作）を行うモジュール

# ---------- 2. LangSmith（LangChain のログ可視化サービス）の設定 ----------
os.environ["LANGCHAIN_TRACING_V2"] = (
    "true"  # LangSmith で詳細ログを取るためのフラグを ON
)
project_name = os.path.splitext(os.path.basename(__file__))[
    0
]  # 実行中ファイル名から拡張子を外して取得
os.environ["LANGCHAIN_PROJECT"] = (
    project_name  # LangSmith 上のプロジェクト名として環境変数に登録
)


# ---------- 3. .mdx ファイルだけを通すフィルタ関数 ----------
def file_filter(file_path: str) -> bool:  # 引数: ファイルのパス（文字列）
    return file_path.endswith(".mdx")  # 戻り値: 拡張子が .mdx のとき True


# ---------- 4. Git リポジトリのクローンとドキュメント読み込み ----------
from langchain_community.document_loaders import (
    GitLoader,
)  # GitHub リポジトリを読み込むクラス

repo_path = "./langchain"  # クローン先フォルダ名
if not os.path.exists(repo_path):  # フォルダが無い＝初回実行と判定
    loder = GitLoader(  # ★変数名はタイプミスだが既存実装を尊重
        clone_url="https://github.com/langchain-ai/langchain",  # 取得元リポジトリの URL
        repo_path=repo_path,  # ローカル保存先
        branch="master",  # 取得するブランチ
        file_filter=file_filter,  # .mdx だけ取り込むフィルタ関数
    )
    documents = loder.load()  # リポジトリをクローンしてドキュメント化
else:  # 既にクローン済みなら更新のみ
    loder = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter,
    )
    documents = loder.load()  # ローカル情報を更新して再読込

print(len(documents))  # 取得できたドキュメント数を標準出力

# ---------- 5. ドキュメントをベクトル化して Chroma に登録 ----------
from langchain_chroma import Chroma  # ベクトルストア実装
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # LLM と埋め込み生成クラス

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 埋め込みモデルを指定
db = Chroma.from_documents(documents, embeddings)  # ドキュメントをベクトル化して保存
retriever = db.as_retriever()  # 検索インタフェースを取得

# ---------- 6. プロンプト・パーサーなど LCEL（LangChain Expression Language）部品 ----------
from langchain_core.output_parsers import StrOutputParser  # 出力を文字列に整形
from langchain_core.prompts import (
    ChatPromptTemplate,
)  # system/human メッセージ用テンプレ
from langchain_core.runnables import RunnablePassthrough  # 値をそのまま渡すノード

# 文脈と質問を差し込むテンプレートを定義
prompt = ChatPromptTemplate.from_template(
    """以下の文脈だけを参考に質問に答えてください。

文脈:
\"\"\"
{context}
\"\"\"

質問: {question}
"""
)

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 温度 0＝同じ入力で同じ出力

from langchain_core.documents import Document


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


from langchain_community.retrievers import BM25Retriever

chroma_retriever = retriever.with_config({"run:name": "chroma_retriever"})

bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run:name": "bm25_retriever"}
)

from langchain_core.runnables import RunnableParallel

hybrid_retriever = (
    RunnableParallel(
        {
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever,
        }
    )
    | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
    | recipocal_rank_fusion
)
hybrid_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hybrid_retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

output = hybrid_rag_chain.invoke("LangChainの概要を教えて")
print(output)
