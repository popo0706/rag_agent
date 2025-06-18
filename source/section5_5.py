# ===============================================================
# 【概要】
# このスクリプトは「LangChain」と「SQLite」を使って、
# 　1. ユーザーと AI のチャット履歴をデータベースに保存しながら
# 　2. その履歴をプロンプトに差し込み
# 　3. OpenAI モデル（gpt-4.1-nano）で回答を生成する
# ……という一連の流れを実装したものです。
# 「respond()」を呼び出せば、過去のやり取りを踏まえた返答が返ります。
# ===============================================================

from uuid import uuid4  # 会話ごとに一意なセッション ID を生成するため
from langchain_community.chat_message_histories import (
    SQLChatMessageHistory,
)  # SQLite に履歴を保存
from langchain_core.prompts import (
    ChatPromptTemplate,
)  # 履歴＋質問をまとめるプロンプトを定義
from langchain_core.output_parsers import (
    StrOutputParser,
)  # LLM 出力（メッセージ形式）→文字列へ
from langchain_openai import ChatOpenAI  # OpenAI Chat API を呼び出すラッパー


# ---------------------------------------------------------------
# 会話の本体：履歴付き応答を返す関数
# ---------------------------------------------------------------
def respond(session_id: str, human_message: str) -> str:
    # ――① 既存のチャット履歴を SQLite から取得 ――――――――――――――――――――
    history = SQLChatMessageHistory(
        session_id=session_id,  # セッションごとに履歴を分ける
        connection="sqlite:///source/session5_5_sql/sqlite.db",  # DB ファイル（自動生成）
    )
    past_messages = history.get_messages()  # これまでの発言を List[BaseMessage] で取得

    # ――② 会話履歴＋今回の質問を差し込むプロンプトを作成 ―――――――――――――――
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは親切なAIです。以下はこれまでの会話履歴です。\n{chat_history}",
            ),
            ("human", "{input}"),
        ]
    )

    # ――③ モデルとチェーンを組み立てる ――――――――――――――――――――――――――
    model = ChatOpenAI(
        model="gpt-4.1-nano", temperature=0
    )  # 温度 0 で determinisitc 出力
    chain = prompt | model | StrOutputParser()  # プロンプト → モデル → 文字列

    # ――④ 履歴を文字列に整形してチェーンを実行 ―――――――――――――――――――――
    chat_history_text = "\n".join(
        f"{m.type}: {m.content}" for m in past_messages  # system/human/ai: 発言内容
    )
    ai_message = chain.invoke(
        {
            "chat_history": chat_history_text,  # {chat_history} に注入
            "input": human_message,  # {input} に注入（今回の質問）
        }
    )

    # ――⑤ 今回の発言を履歴へ追記し、返答を呼び出し元へ返す ―――――――――――――――
    history.add_user_message(human_message)  # ユーザー発言を保存
    history.add_ai_message(ai_message)  # AI 発言を保存

    return ai_message


# ---------------------------------------------------------------
# 動作確認用：このファイルを直接実行したときだけ動くブロック
# ---------------------------------------------------------------
if __name__ == "__main__":
    session_id = uuid4().hex  # 新しいセッション ID を生成
    print(respond(session_id, "こんにちは、私はジョンと言います。"))
    print(respond(session_id, "私の名前がわかりますか？"))
