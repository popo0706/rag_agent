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

# ---------- 7. Retriever を 2 系統用意（LangChain 文書 / Web） ----------
from langchain_community.retrievers import TavilySearchAPIRetriever

langchain_document_retriever = retriever.with_config(  # LangChain 文書用 Retriever
    {"run_name": "langchain_document_retriever"}
)

web_retriever = TavilySearchAPIRetriever(
    k=3
).with_config(  # Web 用 Retriever（上位 3 件取得）
    {"run_name": "Web_retriever"}
)

# ---------- 8. ルーティング判定用データモデル ----------
from enum import Enum  # 列挙型を定義するための標準ライブラリ
from pydantic import BaseModel  # データバリデーション用ライブラリ


class Route(str, Enum):  # Retriever の選択肢を列挙
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):  # LLM から返ってくる構造化出力
    route: Route


# ---------- 9. ルート選択用プロンプトとチェーン ----------
route_prompt = ChatPromptTemplate.from_template(
    """\
    質問に回答するために、適切なRetrieverを選択してください。
    質問:{question}    
    """
)

route_chain = (
    route_prompt
    | model.with_structured_output(RouteOutput)  # 構造化出力として RouteOutput 型を期待
    | (lambda x: x.route)  # 結果から route フィールドだけ取り出す
)

# ---------- 10. ルーティングに応じて Retriever を呼び出す関数 ----------
from typing import Any  # 任意型ヒント用
from langchain_core.documents import Document  # ドキュメント型


def routed_retriever(inp: dict[str, Any]) -> list[Document]:  # 引数: question と route
    question = inp["question"]  # 質問文を取得
    route = inp["route"]  # どの Retriever を使うか

    if route == Route.langchain_document:  # LangChain 文書を使う場合
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:  # Web 検索を使う場合
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route:{route}")  # 想定外の route はエラー


# ---------- 11. ルート判定 → 検索 → プロンプト生成 → 回答生成 の統合チェーン ----------
route_rag_chain = (
    {
        "question": RunnablePassthrough(),  # 質問を次ステップへそのまま渡す
        "route": route_chain,  # まずはルート選択
    }
    | RunnablePassthrough.assign(
        context=routed_retriever
    )  # 文脈を取得してコンテキストに格納
    | prompt  # プロンプトテンプレートを適用
    | model  # LLM で回答生成
    | StrOutputParser()  # 文字列に整形
)

# ---------- 12. チェーンを実行して結果を表示 ----------
output = route_rag_chain.invoke("Langchainの概要を教えて")  # サンプル質問
print(output)  # 生成された回答を表示

print("------------------------------------------------")  # 生成された回答を表示

output2 = route_rag_chain.invoke("東京の今日の天気は")  # サンプル質問
print(output2)  # 生成された回答を表示
