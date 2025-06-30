# =======================================
# ★今回のポイント・用語まとめ★
# ・LangChain / LangGraph …… LLM を「部品（ノード）」として繋ぎ、
#   一連の処理を“チェーン”や“グラフ”として実行できる Python ライブラリ。
# ・LangSmith ………………… LangChain が送受信したプロンプト／レスポンスを
#   クラウド上で可視化・デバッグできるサービス。
# ・Pydantic(BaseModel) …… 入力値の「型」と「必須項目」を自動チェックしてくれる。
# ・ConfigurableField ……… LLM 呼び出し時のパラメータ(max_tokens など)を
#   後から差し替えられる便利な仕組み。
# ・RAG（Retrieval-Augmented Generation）
#   └ 検索(Embedding でベクトル化)＋生成(LLM)のハイブリッド手法。
#     本コードは検索前段の「ロール選定ノード」を担当する。
# =======================================

"""
このスクリプトは「ユーザの質問を見て、どの専門家ロール(1〜3)に答えさせるか」を
GPT-4 に決定してもらう“ロール選定ノード”を中心に実装したサンプルです。

処理の流れ
1. State クラスでチャット全体の状態を保持
2. selection_node でロール番号だけを返させる（max_tokens=1 の強制）
3. answering_node で実際の回答を生成
4. check_node で回答品質を判定し、NG ならロール選定からやり直し

RAG システムの第一段階として「役割振り分け」を体験できます。
"""

# ───────────────────────────────────────────────
#  標準ライブラリ
# ───────────────────────────────────────────────
import os  # 環境変数を読み書きするため

# ───────────────────────────────────────────────
#  LangSmith（プロンプトの送受信をクラウドで可視化）
# ───────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # トレースを有効にする
project_name = os.path.splitext(os.path.basename(__file__))[0]
os.environ["LANGCHAIN_PROJECT"] = project_name  # 例: ファイル名 = プロジェクト名

# ───────────────────────────────────────────────
#  回答ロールの一覧（フロントエンドでも参照できるよう dict で定義）
# ───────────────────────────────────────────────
ROLES = {
    "1": {
        "name": "一般知識エキスパート",
        "description": "幅広分野の一般的な質問に答える",
        "details": "幅広い分野の一般的な質問に対して、正確でわかりやすい回答を提供してください。",
    },
    "2": {
        "name": "生成AI製品エキスパート",
        "description": "生成AIや関連製品、技術に関する専門的な質問に答える",
        "details": "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください。",
    },
    "3": {
        "name": "カウンセラー",
        "description": "個人の悩みや心理的な問題に対してサポートを提供する",
        "details": "個人の悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください。",
    },
}

# ───────────────────────────────────────────────
#  データ構造の定義
# ───────────────────────────────────────────────
import operator  # list を“足し算”するためのヘルパ
from typing import Annotated  # Pydantic と相性の良い型ヒント

from langchain_core.pydantic_v1 import BaseModel, Field  # データ検証ライブラリ


class State(BaseModel):
    """
    チェーン全体で共有する“状態”を 1 つのオブジェクトにまとめるクラス。

    Attributes
    ----------
    query : str
        ユーザが入力した質問文
    current_role : str
        選定されたロール名（例: “一般知識エキスパート”）
    messages : list[str]
        これまでの回答履歴。
        `Annotated[..., operator.add]` を付けると
        「既存リスト + 新しいリスト」を自動でマージする LangChain の裏技が使える。
    current_judge : bool
        check_node での合否。True なら合格。
    judgement_reason : str
        不合格のときの理由を LLM に記述させる。
    """

    query: str = Field(..., description="ユーザからの質問")
    current_role: str = Field(default="", description="選定されたロール")
    messages: Annotated[list[str], operator.add] = Field(
        default=[], description="回答履歴"
    )
    current_judge: bool = Field(default=False, description="品質チェックの結果")
    judgement_reason: str = Field(default="", description="品質チェックの判定理由")


# ───────────────────────────────────────────────
#  LLM の用意 (OpenAI GPT-4)
# ───────────────────────────────────────────────
from langchain_openai import ChatOpenAI  # OpenAI チャットモデルのラッパ
from langchain_core.runnables import ConfigurableField  # 実行時パラメータ差し替え

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)  # 温度0で“ブレ”を抑える

# max_tokens を外から上書きできるように登録
# selection_node では「1 トークンに固定」して番号のみを出力させる。
llm = llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))

# ───────────────────────────────────────────────
#  ノード定義（LangGraph で「箱」として扱われる関数たち）
# ───────────────────────────────────────────────
from typing import Any

from langchain_core.prompts import ChatPromptTemplate  # プロンプトテンプレート
from langchain_core.output_parsers import StrOutputParser  # 出力→文字列


def selection_node(state: State) -> dict[str, Any]:
    """
    質問文を GPT-4 に渡し、“どのロールが最適か”を
    1, 2, 3 の数字だけで返してもらうノード。

    Parameters
    ----------
    state : State
        現在のチャット状態（ここでは state.query を使う）

    Returns
    -------
    dict[str, Any]
        {"current_role": <ロール名>}
        LangGraph では「戻り値の dict が State に自動マージ」される。
    """
    query = state.query

    # 例）「1.一般知識エキスパート:幅広分野…」のような選択肢を作る
    role_options = "\n".join(
        [f"{k}.{v['name']}:{v['description']}" for k, v in ROLES.items()]
    )

    # ----- プロンプト -----
    prompt = ChatPromptTemplate.from_template(
        """
        質問を分析し、最も適切な回答担当ロールを選択してください。

        選択肢：
        {role_options}

        回答は選択肢の番号(1、2、または3)のみを返してください。

        質問：{query}
        """.strip()
    )

    # max_tokens=1 で「数字一文字しか返せない」ように縛る
    chain = (
        prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    )

    # たとえば "2" のような結果を想定
    role_number = chain.invoke({"role_options": role_options, "query": query})

    # strip() で余分な空白や改行を除去し、正式なロール名へ変換
    selected_role = ROLES[role_number.strip()]["name"]

    # State に current_role として保存される
    return {"current_role": selected_role}

    # ※ 以下の重複ブロックは元コード維持のため残しています
    chain = (
        prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
    )
    role_number = chain.invoke({"role_options": role_options, "query": query})
    selected_role = ROLES[role_number.strip()]["name"]
    return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
    """
    選ばれたロールになりきって実際の回答を生成するノード。
    """
    query = state.query
    role = state.current_role

    # role_details は “各ロールが何をするのか” を LL M に思い出させる説明
    role_details = "\n".join([f"-{v['name']}:{v['details']}" for v in ROLES.values()])

    prompt = ChatPromptTemplate.from_template(
        """
        あなたは{role}として回答してください。以下の質問に対して、あなたの役割に基づいた適切な回答を提供してください。

        役割の詳細：
        {role_details}

        質問：{query}

        回答：
        """.strip()
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"role": role, "role_details": role_details, "query": query})
    return {"messages": [answer]}  # State.messages に追記される


class Judgement(BaseModel):
    """
    回答チェック用の構造化出力。

    reason : なぜ OK / NG と判定したか
    judge  : True なら合格、False なら不合格
    """

    reason: str = Field(default="", description="判定理由")
    judge: bool = Field(default=False, description="判定結果")


def check_node(state: State) -> dict[str, Any]:
    """
    回答の品質を GPT-4 に自己評価させるノード。
    合格ならワークフローを終了、不合格なら selection_node に戻す。
    """
    query = state.query
    answer = state.messages[-1]  # 直近の回答

    prompt = ChatPromptTemplate.from_template(
        """
        以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を回答してください。
        また、その判断理由も説明してください。

        ユーザからの質問：{query}
        回答：{answer}
        """.strip()
    )

    # with_structured_output を使うと Pydantic モデルで受け取れる
    chain = prompt | llm.with_structured_output(Judgement)
    result: Judgement = chain.invoke({"query": query, "answer": answer})

    return {"current_judge": result.judge, "judgement_reason": result.reason}


# ───────────────────────────────────────────────
#  グラフの組み立て
# ───────────────────────────────────────────────
from langgraph.graph import StateGraph  # 有向グラフを構築
from langgraph.graph import END  # “終了”ノード

workflow = StateGraph(State)

# ノードを登録
workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

# スタート地点
workflow.set_entry_point("selection")

# 直線的なエッジ
workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

# check_node の結果によって分岐
workflow.add_conditional_edges(
    "check",
    lambda state: state.current_judge,  # True / False を判定
    {True: END, False: "selection"},  # 合格→終了、不合格→ロール再選定
)

# グラフを「実行可能オブジェクト」に変換
compiled = workflow.compile()

# ───────────────────────────────────────────────
#  動作テスト
# ───────────────────────────────────────────────
initial_state = State(query="生成AIについて教えてください。")
result = compiled.invoke(initial_state)

print(result)
print("---------------------------------------------------")
print(result["messages"][-1])
