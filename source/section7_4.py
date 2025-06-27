# ------------------------------------------------------------------------------
# 【ソースコード全体の要約】
# GitHub 上の LangChain リポジトリをクローンし、拡張子 .mdx のみを読み込んで
# ドキュメントを作成します。初回のみネットワークから取得し、２回目以降は
# pickle で保存したキャッシュを利用します。その後 ragas を使ってドキュメント
# からテストセットを自動生成し、LangSmith にアップロードします。
#
# 《今回の学習ポイント》
# ・「キャッシュ」：ネットワークアクセスを減らし高速化／安定化する実践テクニック
# ・「ファイルフィルタ」：対象ファイルを拡張子で絞り込む関数の書き方
# ・「外部サービス連携」：OpenAI や LangSmith など複数サービスの API を一気通貫
#   で扱う際の流れ（認証 → オブジェクト生成 → 実行 → エラーハンドリング）
# ・「pickle の使いどころと注意点」：手軽だがセキュリティ面では要注意
# ・「逐次処理 vs 非同期処理」：Jupyter 等でイベントループが競合しないように
#   nest_asyncio.apply() を使ってパッチを当てる理由
# ------------------------------------------------------------------------------

"""
【概要】
このスクリプトは
① GitHub 上にある LangChain リポジトリをローカルへクローンし、
② 拡張子 .mdx のファイルのみを LangChain の GitLoader で読み込み、
③ 取得したドキュメントを pickle でキャッシュ保存し、
④ 次回実行時はキャッシュを再利用してネットワークアクセスを省き、
⑤ 最終的にドキュメント数を表示し、その後 ragas を用いてテストセットを自動生成します。
"""

# ------------------------------------------------------------------------------
# 標準ライブラリを読み込む（ファイル操作やオブジェクトのシリアライズに使用）
# ------------------------------------------------------------------------------
import os  # OS に依存したパス操作やファイル存在確認などを行うモジュール
import pickle  # Python オブジェクトをバイナリ形式で保存／復元するための標準モジュール

# ------------------------------------------------------------------------------
# 外部ライブラリ（LangChain 公式実装のドキュメントローダー）を読み込む
# ------------------------------------------------------------------------------
from langchain_community.document_loaders import (
    GitLoader,
)  # Git リポジトリからファイルをロードするユーティリティ


# ------------------------------------------------------------------------------
# 拡張子 .mdx のみを読み込むためのフィルタ関数
# GitLoader は file_filter 引数にコールバック関数を受け取る設計なので、
# True が返ったファイルだけがロード対象になります。
# ------------------------------------------------------------------------------
def file_filter(
    file_path: str,
) -> bool:  # GitLoader が渡してくるパスを受け取り、bool を返す
    return file_path.endswith(
        ".mdx"
    )  # .mdx で終わるなら True（対象ファイル）、それ以外は False


# ------------------------------------------------------------------------------
# pickle キャッシュファイルの保存場所を定義
# ファイル名は自由だが拡張子 .pkl が慣例的
# ------------------------------------------------------------------------------
cache_path = "document.pkl"  # 同ディレクトリ直下に保存／読み込みを行う

# ------------------------------------------------------------------------------
# キャッシュが存在するか確認し、なければ GitHub からデータを取得
# ------------------------------------------------------------------------------
if not os.path.exists(cache_path):  # ファイルの有無をチェック（True:存在しない）
    # GitLoader のインスタンスを作成（clone_url は公開リポジトリの URL）
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",  # クローン先 Git URL
        repo_path="./langchain",  # ローカルにクローンするパス
        branch="master",  # 利用するブランチ名
        file_filter=file_filter,  # 先ほど定義したフィルタ関数を渡す
    )
    documents = (
        loader.load()
    )  # 実際にリポジトリをクローンし、フィルタ条件でドキュメントを読み込む
    # ↓ 読み込んだドキュメントをバイナリファイルにシリアライズして保存
    with open(cache_path, "wb") as f:  # "wb": write binary モードで開く
        pickle.dump(documents, f)  # オブジェクトをファイルに保存
else:
    # キャッシュが存在する場合はこちらが実行される
    with open(cache_path, "rb") as f:  # "rb": read binary モードで開く
        documents = pickle.load(f)  # ファイルからオブジェクトを逆シリアライズして取得

print(len(documents))  # 取得（または復元）したドキュメント数を標準出力に表示

# ------------------------------------------------------------------------------
# GitLoader が付与したメタデータを補完
# 原則として metadata["source"] にファイルパスが入っているので、
# LangSmith での後処理が楽になるよう "filename" キーにもコピーしておく
# ------------------------------------------------------------------------------
for document in documents:  # documents は LangChain の Document オブジェクトのリスト
    document.metadata["filename"] = document.metadata["source"]  # 同じ値を別キーに格納

# ------------------------------------------------------------------------------
# 以降は ragas を使ってテストセットを生成する準備
# ------------------------------------------------------------------------------
import nest_asyncio  # Jupyter 上で asyncio のイベントループ競合を回避するパッチ
from ragas.testset.generator import TestsetGenerator  # ragas のコアクラス
from ragas.testset.evolutions import (
    simple,
    reasoning,
    multi_context,
)  # 質問生成の進化プロンプト
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI の LLM & 埋め込み

# Jupyter など既にイベントループが走っている環境でもエラーを出さないようにパッチを当てる
nest_asyncio.apply()

# ------------------------------------------------------------------------------
# TestsetGenerator を LLM と Embedding を指定して生成
# generator_llm と critic_llm はテスト生成と評価で同じモデルを使う例
# ------------------------------------------------------------------------------
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4.1-nano"),  # 質問生成に使う LLM
    critic_llm=ChatOpenAI(model="gpt-4.1-nano"),  # 質問の質を評価する LLM
    embeddings=OpenAIEmbeddings(),  # コンテキスト検索用の埋め込みモデル
)

# ------------------------------------------------------------------------------
# ドキュメント群からテストセットを生成
# test_size=4 なので 4 問を作成
# distributions で生成タイプの比率を調整
# raise_exceptions=True にすると途中失敗時に即例外を投げる（デバッグしやすい）
# ------------------------------------------------------------------------------
testset = generator.generate_with_langchain_docs(
    documents,  # 対象のドキュメント
    test_size=4,  # 生成するテストの総数
    distributions={
        simple: 0.5,  # 50% はシンプルな単一文脈問題
        reasoning: 0.25,  # 25% は推論を要する問題
        multi_context: 0.25,  # 25% は複数文脈問題
    },
    raise_exceptions=True,  # 例外を上位に伝搬させて失敗を検知しやすく
)

# pandas に変換してコンソールで確認（デバッグ目的）
df = testset.to_pandas()
print(df)

# ------------------------------------------------------------------------------
# 生成したテストセットを LangSmith（データセット管理プラットフォーム）にアップロード
# ------------------------------------------------------------------------------
from langsmith import Client  # LangSmith の Python SDK

dataset_name = "agent-book"  # LangSmith 上でのデータセット名

client = Client()  # 認証情報は環境変数 LANGCHAIN_API_KEY 等で自動検出される

# 既に同名データセットがある場合は削除して上書き
if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)

# 新しい空のデータセットを作成
dataset = client.create_dataset(dataset_name=dataset_name)

# ------------------------------------------------------------------------------
# TestsetRecord オブジェクトを LangSmith の「Example」形式に変換するための準備
# ------------------------------------------------------------------------------
inputs = []  # LangSmith では "inputs" と "outputs" のスキーマが固定
outputs = []
metadatas = []  # 任意の付随情報（検索時などに便利）

# 各テストレコードをループし、LangSmith のフォーマットへ落とし込む
for testset_record in testset.test_data:
    # 入力はユーザーからの質問のみ
    inputs.append({"question": testset_record.question})
    # 出力はモデルが返すべきコンテキストと正解
    outputs.append(
        {
            "contexts": testset_record.contexts,
            "ground_truth": testset_record.ground_truth,
        }
    )
    # メタデータにはソースファイルと問題タイプを保存
    metadatas.append(
        {
            "source": testset_record.metadata[0]["source"],
            "evolution_type": testset_record.evolution_type,
        }
    )

# ------------------------------------------------------------------------------
# まとめて LangSmith にアップロード（バルク作成）
# dataset_id は create_dataset の戻り値に含まれる
# ------------------------------------------------------------------------------
client.create_examples(
    inputs=inputs,
    outputs=outputs,
    metadata=metadatas,
    dataset_id=dataset.id,
)
