# ------------------------------------------------------------------------------
# 🔰 今回のポイント・用語まとめ
# ・RAG（Retrieval-Augmented Generation）
#     └ 検索で取り出したテキスト（外部知識）を LLM の追加コンテキストにして回答精度を上げる手法
# ・キャッシュ
#     └ ネットワークや API コールの回数を減らすために、一度取り出したデータをローカルに保存する仕組み
# ・pickle
#     └ Python オブジェクトをそのままバイナリ化して保存／復元できる標準モジュール（※信頼できるデータのみで）
# ・LangChain / LangSmith
#     └ LangChain は LLM ワークフローの OSS、LangSmith はその実験管理 SaaS
# ・ragas
#     └ ドキュメント QA 用のテストセットを自動生成してくれるライブラリ
# ------------------------------------------------------------------------------

"""
GitHub 上の LangChain リポジトリから `.mdx` だけを読み込み、
データを pickle でキャッシュしつつ ragas で QA テストセットを作成し、
最終的に LangSmith へアップロードするスクリプト。
初回実行はネットワークアクセスが発生し、２回目以降はキャッシュを再利用する。
ドキュメント QA（＝RAG システム）の評価データづくりを自動化できる。
"""

# ===== ここからインポート =====
# OS 依存のパス操作やファイル存在チェック
import os

# Python オブジェクトをバイナリ形式で保存／復元する
import pickle

# Git リポジトリからドキュメントを読み込む LangChain コミュニティ実装
from langchain_community.document_loaders import GitLoader

# Jupyter などで既に走っている asyncio ループと競合しないようにするパッチ
import nest_asyncio

# ragas のテストセット作成ツール
from ragas.testset.generator import TestsetGenerator

# テスト問題の “種類” を作るための進化プロンプト
from ragas.testset.evolutions import simple, reasoning, multi_context

# OpenAI のチャット LLM と埋め込みモデル
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangSmith へデータを送る SDK
from langsmith import Client

# ===== インポート終わり =====


# ------------------------------------------------------------------------------
# 拡張子が .mdx のファイルだけを取り込むためのフィルタ関数
# GitLoader に callback として渡すと、True を返すファイルのみロードされる
# ------------------------------------------------------------------------------
def file_filter(file_path: str) -> bool:
    """
    .mdx ファイルだけを選び出すコールバック

    Parameters
    ----------
    file_path : str
        GitLoader から渡ってくるファイルのフルパス

    Returns
    -------
    bool
        拡張子が .mdx なら True（= 取り込む）／それ以外は False

    Notes
    -----
    GitLoader 側が「この関数が True を返したファイルだけ読む」という設計。
    拡張子チェックは endswith で十分軽量。(参考: str.endswith)
    """
    return file_path.endswith(".mdx")


# ------------------------------------------------------------------------------
# pickle のキャッシュファイルを置くパス
# ------------------------------------------------------------------------------
cache_path = "document.pkl"

# ------------------------------------------------------------------------------
# ① キャッシュが無ければ GitHub から clone → 読み込み → pickle 保存
# ② あれば pickle から即時ロード
# ------------------------------------------------------------------------------
if not os.path.exists(cache_path):
    # GitLoader は clone → checkout → ファイル走査まで一括でやってくれる
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",  # clone 先のローカルフォルダ
        branch="master",
        file_filter=file_filter,  # .mdx だけ読むように設定
    )
    # ネットワーク I/O が発生する重い処理
    documents = loader.load()

    # 一度取得したら pickle で保存して次回から高速化
    with open(cache_path, "wb") as f:
        # なぜ wb か? ― バイナリモードで書き込み(write binary)するため
        pickle.dump(documents, f)
else:
    # キャッシュ利用時は一瞬でロード完了
    with open(cache_path, "rb") as f:
        documents = pickle.load(f)

print(len(documents))  # どのくらいのドキュメントを扱うか可視化しておく


# ------------------------------------------------------------------------------
# LangSmith で後処理しやすいよう、metadata に 'filename' キーを足す
# （メタデータを後で検索・フィルタしたいときに便利）
# ------------------------------------------------------------------------------
for document in documents:
    # GitLoader が付与済みの "source" を "filename" にコピーするだけ
    document.metadata["filename"] = document.metadata["source"]


# ------------------------------------------------------------------------------
# ragas でテストセットを作る準備
# Jupyter でネストしたイベントループがあると怒られるので先にパッチ
# ------------------------------------------------------------------------------
nest_asyncio.apply()

# ------------------------------------------------------------------------------
# TestsetGenerator の構築
# ─ generator_llm : テスト問題（質問）を生成する LLM
# ─ critic_llm    : 質の低い質問を弾く LLM
# ─ embeddings    : コサイン類似度検索に使う埋め込みモデル
# ------------------------------------------------------------------------------
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4.1-nano"),
    critic_llm=ChatOpenAI(model="gpt-4.1-nano"),
    embeddings=OpenAIEmbeddings(),
)

# ------------------------------------------------------------------------------
# ドキュメントを食わせてテストセットを自動生成
# distributions で「どのタイプの問題を何割つくるか」を指定している
# ------------------------------------------------------------------------------
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=4,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    raise_exceptions=True,  # デバッグしやすいよう途中失敗で即例外
)

# DataFrame で中身をざっと確認 (pandas に変換するメソッドが標準装備)
df = testset.to_pandas()
print(df)


# ------------------------------------------------------------------------------
# LangSmith へアップロードする
# LangSmith は「入力(inputs)」「出力(outputs)」「メタデータ(metadata)」の
# ３カラムで Examples を管理するため、まずそれぞれのリストを作る
# ------------------------------------------------------------------------------
dataset_name = "agent-book"
client = Client()  # 環境変数 LANGCHAIN_API_KEY 等で自動認証

# 同名データセットが既にあれば削除して作り直し（重複防止）
if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)
dataset = client.create_dataset(dataset_name=dataset_name)

# LangSmith 用にフォーマットを詰め替える
inputs, outputs, metadatas = [], [], []
for record in testset.test_data:
    # モデルへの入力は「質問」のみ
    inputs.append({"question": record.question})

    # モデルが返すべき答え（コンテキストと正解）を outputs に
    outputs.append(
        {
            "contexts": record.contexts,
            "ground_truth": record.ground_truth,
        }
    )

    # 追加情報としてソースファイルと問題タイプをメタデータへ
    metadatas.append(
        {
            "source": record.metadata[0]["source"],
            "evolution_type": record.evolution_type,
        }
    )

# 一括で Example を登録
client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id,
)
