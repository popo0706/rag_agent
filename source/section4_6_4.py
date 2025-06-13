# ==============================================================
# ❶ LangChain の公式 GitHub リポジトリをクローンし、
# ❷ .mdx ドキュメントだけ抽出して 1,000 文字ごとに分割（チャンク化）し、
# ❸ OpenAI Embeddings（text-embedding-3-small）でベクトル化して
# ❹ Chroma ベクトルストアへ登録、
# ❺ Retriever として公開し、サンプルクエリを投げて
# ❻ 1 件目の検索結果を表示する――という一連のデモスクリプトです。
# ==============================================================


# ---------- 0. 必要ライブラリの import ---------------------------------
#   ここでは「ベクトルストア／Embeddings／ドキュメントローダ」という
#   主要 3 コンポーネントを読み込む。
#   ─────────────────────────────────
from langchain_chroma import Chroma  # Chroma = ベクトルストア実装
from langchain_openai import (
    OpenAIEmbeddings,
)  # OpenAI の埋め込みモデル（ラッパークラス）
from langchain_community.document_loaders import (  # LangChain コミュニティ提供のローダ集
    GitLoader,  #   └ Git リポジトリを丸ごと読み込むローダ
)


# ---------- 1. .mdx のみ通すフィルタ関数 --------------------------------
def file_filter(file_path: str) -> bool:
    """
    GitLoader が走査する「ファイルパス」を受け取り、
    拡張子が `.mdx` の場合のみ True を返す。
    ※ True を返したファイルだけがローダに取り込まれる。
    """
    return file_path.endswith(".mdx")  # 末尾判定だけのシンプル実装


# ---------- 2. Git からドキュメント取得 --------------------------------
#   GitLoader は以下 4 つのパラメータで挙動が決まる：
#   ① clone_url  : 取得元リポジトリ（HTTPS URL でも OK）
#   ② repo_path  : ローカルにクローンするパス
#   ③ branch     : 対象ブランチ（main / develop など）
#   ④ file_filter: 取得対象ファイルを制限するコールバック
loader = GitLoader(
    clone_url="http://github.com/langchain-ai/langchain",  # ① 取得元 URL
    repo_path="./langchain",  # ② ローカル保存先
    branch="master",  # ③ 対象ブランチ
    file_filter=file_filter,  # ④ 拡張子フィルタ
)

raw_docs = loader.load()  # Git から取得した Document オブジェクトのリスト


# ---------- 3. 1,000 文字ごとにチャンク化 ------------------------------
#   “巨大なドキュメント” を “検索しやすい粒度” に割く作業。
#   LangChain では TextSplitter 系のユーティリティが豊富。
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # 1 チャンク＝1,000 文字
    chunk_overlap=0,  # チャンク間の重複なし（今回は学習用途なので 0）
)

docs = text_splitter.split_documents(raw_docs)  # チャンク後のドキュメント一覧


# ---------- 4. OpenAI Embeddings でベクトル化 ---------------------------
#   text-embedding-3-small = 1,536 次元の軽量モデル。
#   高精度が欲しい場合は *-large を選択すると次元数が増える（＝計算コスト増）。
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1,536 次元ベクトル


# ---------- 5. Chroma へ登録 -------------------------------------------
#   Chroma は in-process DB（DuckDB + Parquet）なのでローカル実行が手軽。
#   from_documents() が「ドキュメント＋エンベッディング」をまとめて登録。
#   ★注意: コメントの誤 typo `from_documnets` は残しておく（あくまでコメント）。
db = Chroma.from_documents(docs, embeddings)  # ベクトルストアへ一括登録


# ---------- 6. Retriever 化 --------------------------------------------
#   Chroma オブジェクトに as_retriever() を噛ませると
#   LangChain 標準の “Retriever インターフェース” に早変わり。
retriever = db.as_retriever()  # 類似検索インターフェースを取得


# ---------- 7. チェーン組み立て & 実行例 -------------------------------
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

query = "AWSのs3からデータを読み込むためのDocumentloaderはありますか？"  # 検索クエリ例

# ■ Prompt Engineering
#   {context} には Retriever が返す類似ドキュメント群を、
#   {question} にはユーザー入力（この場合は query）を流し込む。
prompt = ChatPromptTemplate.from_template(
    '''
    文脈:
    """
    {context}
    """
    質問:
    """
    {question}
    """
    '''
)

# ■ Chat Model 準備（今回は gpt-4.1-nano を使用）
model = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)  # 0 = 完全決定論的

from langchain_core.output_parsers import StrOutputParser  # 出力を str に変換
from langchain_core.runnables import RunnablePassthrough  # 値をそのまま流すパススルー


# ■ Runnable チェーン定義
#   ① Retriever で文脈作成 → ② Prompt 成形 → ③ LLM 推論 → ④ 文字列抽出
chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # ①
    | prompt  # ②
    | model  # ③
    | StrOutputParser()  # ④
)

output = chain.invoke(query)  # チェーン実行
print(output)  # 検索結果（最初の回答）を表示
