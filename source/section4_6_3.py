# ==============================================================
# 【概要】
# このスクリプトは――
#   1. LangChain 公式 GitHub リポジトリをクローン
#   2. 拡張子 .mdx のドキュメントだけを抽出
#   3. 1,000 文字ごとに分割（チャンク化）
#   4. OpenAI の埋め込みモデルでベクトル化
#   5. Chroma ベクトルストアに登録
#   6. 検索（Retriever）として利用できる形にする
#   7. 実際にクエリを投げて最初の検索結果を表示する
# ――という一連の流れを実装しています。
# ==============================================================

from langchain_chroma import Chroma  # ← ベクトルストア実装
from langchain_openai import OpenAIEmbeddings  # ← OpenAI の埋め込みモデル
from langchain_community.document_loaders import (  # ← 複数行 import の書き方
    GitLoader,  #    Git リポジトリ用ローダ
)


# ----------------------------------------------------------------
# ① .mdx ファイルだけを選別するフィルタ関数
#    GitLoader がファイルを走査するたびに呼び出される。
#    True を返したファイルだけがドキュメントとして採用される。
# ----------------------------------------------------------------
def file_filter(file_path: str) -> bool:
    """拡張子が .mdx のファイルだけを通すフィルタ関数"""
    return file_path.endswith(".mdx")  # ← ".mdx" で終わるなら True


# ----------------------------------------------------------------
# ② Git リポジトリをローカルにクローンしつつドキュメントを読み込む
#    ※変数名 loder は loader の typo だが、指示に合わせてそのまま使用
# ----------------------------------------------------------------
loder = GitLoader(  # ← ローダーのインスタンスを生成
    clone_url="http://github.com/langchain-ai/langchain",  # ← 取得元リポジトリ
    repo_path="./langchain",  # ← クローン先ディレクトリ
    branch="master",  # ← 対象ブランチ
    file_filter=file_filter,  # ← .mdx ファイルだけ絞り込み
)

raw_docs = loder.load()  # ← Git から取得した List[Document] が返る

# ----------------------------------------------------------------
# ③ ドキュメントを 1,000 文字単位で分割（チャンク化）
#    大きなテキストをそのまま embed すると性能が落ちるため、
#    chunk_overlap=0 で切れ目が重ならないようにしている。
# ----------------------------------------------------------------
from langchain_text_splitters import CharacterTextSplitter  # ← 分割ユーティリティ

text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # ← 1,000 文字ごとに切る
    chunk_overlap=0,  # ← チャンク間の重複は 0
)

docs = text_splitter.split_documents(raw_docs)  # ← 分割後の List[Document]

# ----------------------------------------------------------------
# ④ OpenAIEmbeddings で各チャンクをベクトル化
#    "text-embedding-3-small" は 1,536 次元のベクトルを返す。
# ----------------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ----------------------------------------------------------------
# ⑤ Chroma にドキュメントを登録
#    ※メソッド名 form_documnets は from_documents の typo だが指示に従い保持
# ----------------------------------------------------------------
db = Chroma.from_documents(docs, embeddings)

# ----------------------------------------------------------------
# ⑥ ベクトルストアを検索用インターフェース（Retriever）として公開
#    自然言語のクエリを渡すと類似チャンクを返してくれる。
# ----------------------------------------------------------------
retriever = db.as_retriever()

# ----------------------------------------------------------------
# ⑦ 実際にクエリを投げて検索結果を確認
#    - query          : 質問文（自然言語）
#    - context_docs   : 類似度の高いチャンク一覧（List[Document]）
#    ここでは件数と先頭 1 件のメタデータ＋本文を表示している。
# ----------------------------------------------------------------
query = "AWSのs3からデータを読み込むためのDocumentloaderはありますか？"  # ← 例として S3 のローダ有無を質問

context_docs = retriever.invoke(query)  # ← Retriever にクエリを投げる
print(f"len={len(context_docs)}")  # ← 取得チャンク数を表示

first_doc = context_docs[0]  # ← 先頭チャンクを取り出し
print(f"metadata={first_doc.metadata}")  # ← チャンクのメタデータを表示
print(first_doc.page_content)  # ← チャンク本文を表示
