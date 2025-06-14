from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

# 環境変数をロード
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ .env または secrets.toml に OPENAI_API_KEY を設定してください")
    st.stop()

client = OpenAI(api_key=api_key)

# Streamlit のページ設定
st.set_page_config(page_title="ChatGPT 風チャット", page_icon="💬", layout="wide")

# サイドバー：設定オプション
with st.sidebar:
    st.header("⚙️ 設定")
    AVAILABLE_MODELS = [
        "o3",
        "o3-mini",
        "gpt-3.5-turbo",
    ]
    model_name = st.selectbox("使用モデル", AVAILABLE_MODELS, index=0)
    temperature = st.slider("温度 (創造性)", 0.0, 1.0, 0.7, 0.05)

# セッションステートに会話履歴を初期化
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "あなたは優秀なアシスタントです。"}
    ]

st.title("💬 ChatGPT 風チャット")
user_input = st.text_input("メッセージを入力...", key="user_input")
send = st.button("送信", use_container_width=True)
clear = st.button("🗑️ クリア", use_container_width=True)

# クリアボタン押下時は履歴を初期化してリロード
if clear:
    st.session_state["messages"] = st.session_state["messages"][:1]
    st.experimental_rerun()

# メッセージ送信処理
if send and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})

    assistant_response = ""
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=st.session_state["messages"],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            assistant_response += token
        # ストリーミングによる個別表示は削除し、下部の履歴表示ループで全体を表示します
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        assistant_response = "エラーが発生しました。"

    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_response}
    )

# 会話履歴の表示（system メッセージを除く）
for msg in st.session_state["messages"][1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
