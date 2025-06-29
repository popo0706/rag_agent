"""
============================================================
RAG の評価メトリックを試すためのサンプルスクリプト
------------------------------------------------------------
1. 「Run（実行結果）」と「Example（正解データ）」を入力とし、
   RAGas で提供されるメトリックを使ってスコアを計算します。
2. デモ用に常にスコア 1 を返す my_evaluator と、
   RagasMetricEvaluator クラスの 2 つの評価器を実装しています。
3. LLM や Embeddings を必要とするメトリックへはラッパーを介して
   LangChain から RAGas 形式へ自動変換します。
============================================================

【今回のポイント・用語まとめ】
・RAG (Retrieval-Augmented Generation)
    → 質問に答える前に “検索” で外部文書(コンテキスト)を取得し、
      その情報を使って生成(Generation)を行う仕組み。
・Run / Example
    → LangSmith が採点時に扱うデータ構造。
      Run  : 実際のモデル出力（answer, contexts など）
      Example : その問題の正解（ground_truth など）
・MetricWithLLM / MetricWithEmbeddings
    → 「LLM を内部で呼び出すタイプのメトリック」や
      「ベクトル類似度を計算するタイプのメトリック」を区別するための基底クラス。
・LangchainLLMWrapper / LangchainEmbeddingsWrapper
    → LangChain のオブジェクトを RAGas の規格へ噛み合わせるアダプタ。
(参考) RAGas GitHub  https://github.com/explodinggradients/ragas
"""

# ------------------------------------------------------------
# 1. デモ用の evaluator ----------------------------------------
# ------------------------------------------------------------
from typing import Any  # どんな型でも表せる「万能」な型ヒントを提供する標準ライブラリ

# ---- 外部サービス(LangSmith)が定義する型 ---------------------
from langsmith.schemas import (
    Run,
    Example,
)  # 評価時に渡される “Run(実行結果)” と “Example(正解)” の型


def my_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """
    いつ呼んでも固定値のスコア 1 を返す非常に単純な評価関数。
    教材目的で「評価器は dict を返せば良い」という型例を示す。

    Parameters
    ----------
    run : Run
        モデルから実際に得られた出力(answers, contexts など)を
        含むオブジェクト。
    example : Example
        問題文(question)と正解(ground_truth)を含むオブジェクト。

    Returns
    -------
    dict[str, Any]
        "key": 指標名, "score": スコア値 の 2 要素を持つ辞書。
    """
    # 本来は run / example を読んでスコア計算を行うが、
    # デモとして常に 1 を返す。
    return {"key": "sample_metric", "score": 1}


# ------------------------------------------------------------
# 2. RagasMetricEvaluator で使うライブラリの import -------------
# ------------------------------------------------------------
from langchain_core.embeddings import (
    Embeddings,
)  # テキストをベクトル(数値配列)に変換する共通インターフェース
from langchain_core.language_models import (
    BaseChatModel,
)  # チャット形式で LLM を呼び出すための抽象基底クラス

# ------- RAGas 側のラッパー / メトリック ---------------------
from ragas.embeddings import (
    LangchainEmbeddingsWrapper,
)  # LangChain の Embeddings → RAGas 形式へ包むクラス
from ragas.llms import LangchainLLMWrapper  # LangChain の LLM → RAGas 形式へ包むクラス
from ragas.metrics.base import (
    Metric,
    MetricWithEmbeddings,
    MetricWithLLM,
)  # メトリックの基底クラス群


# ------------------------------------------------------------
# 3. RagasMetricEvaluator クラス -------------------------------
# ------------------------------------------------------------
class RagasMetricEvaluator:
    """
    指定された RAGas メトリックで Run を評価し、スコアを返すクラス。

    RAGas のメトリックは以下 3 種類に大別される。
      1. Metric                : 追加リソース不要
      2. MetricWithLLM         : 内部で LLM を使用
      3. MetricWithEmbeddings  : 内部でベクトル距離(例: コサイン類似度)を使用

    このクラスは Metric がどの型かを動的に判定し、
    必要な依存(LLM / Embeddings)をラップして注入する。

    Parameters
    ----------
    metric : Metric
        RAGas が提供する任意の評価指標オブジェクト。
    llm : BaseChatModel
        LangChain で構築した任意のチャット LLM。
        Metric が LLM を必要としない場合はダミーでも可。
    embeddings : Embeddings
        LangChain で構築した埋め込みモデル。
        Metric が Embeddings を必要としない場合はダミーでも可.
    """

    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric  # 評価に使うメトリックを保持

        # Metric が LLM を必要とする場合はラップしてセット
        if isinstance(self.metric, MetricWithLLM):
            # RAGas 側が認識できるよう LangChain → RAGas 形式へ変換
            self.metric.llm = LangchainLLMWrapper(llm)

        # Metric が Embeddings を必要とする場合はラップしてセット
        if isinstance(self.metric, MetricWithEmbeddings):
            # 同様に Embeddings もラップする
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        """
        Run / Example から必要な情報を取り出し、メトリックで採点する。

        Returns
        -------
        dict[str, Any]
            "key": メトリック名, "score": 数値スコア
        """
        # LangChain では contexts が Document 型で格納されることが多い。
        # metric.score() は「ただの文字列リスト」を期待しているため、
        # page_content だけを抽出してリストに変換する。
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # RAGas の metric.score() は質問・回答・コンテキスト・正解を
        # 辞書で受け取る仕様。キー名はライブラリ側で固定されている。
        score = self.metric.score(
            {
                "question": example.inputs["question"],
                "answer": run.outputs["answer"],
                "contexts": context_strs,
                "ground_truth": example.outputs["ground_truth"],
            }
        )

        # 評価結果を LangSmith が期待する形式(dict)で返す
        return {
            "key": self.metric.name,  # 例: "context_precision" など
            "score": score,
        }
