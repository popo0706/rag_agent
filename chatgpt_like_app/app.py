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

ROLE_MAP = {
    "通常の質問": "AAA",
    "ソースコードコメント": "BBB",
    "エラー調査": "CCC",
}

# サイドバー：設定オプション

# サイドバー：設定オプション
with st.sidebar:
    st.header("⚙️ 設定")
    AVAILABLE_MODELS = [
        "o3",
        "o3-mini",
        "gpt-4.1-nano",
    ]
    model_name = st.selectbox("使用モデル", AVAILABLE_MODELS, index=0)
    if model_name not in ["o3", "o3-mini"]:
        temperature = st.slider("温度 (創造性)", 0.0, 1.0, 0.7, 0.05)
    else:
        temperature = None

    st.subheader("システムロール")
    ROLE_NAMES = ["通常の質問", "ソースコードコメント", "エラー調査"]
    role_name = st.selectbox("システムロール", ROLE_NAMES, index=0)

ROLE_MAP = {
    "通常の質問": "あなたは優秀なアシスタントです。これから質問をしますので、中学生でも理解できるように丁寧に、回答が長くてもいいのでしっかりと説明してください。",
    "ソースコードコメント": """
        # ソースコードコメント追加
        # 命令
            -あなたは一流の開発者で、開発PJのリーダです。
            -これから開発知識ゼロの新人がPJに参画するため、これから添付する研修用のソースコードに丁寧なコメントを記述してください。
        # 前提
            -元のソースコードの実装部分は変更しないでください。
        # 条件
            - 開発知識ゼロの新人のため、メソッド単位などではなく、一行ごとに丁寧なコメントを記載してください。
            - 難しそうな箇所、間違いそうな箇所などについてはより丁寧なコメントを記載してください。
            - 元のコメントについては間違っている可能性が高く、一度コメントをすべて削除したうえで実行してください。
            - 大事なことなので繰り返しますが、元のコメントについては間違っている可能性が高く、一度コメントをすべて削除したうえで実行してください。
            - ソースコードの先頭にソースコードの要約説明を記載してください。そのうえで各行にコメントを追加してください。
            - ソースコードの先頭にソースコードの要約説明の中に、'''今回の学習'''ポイントとしてあなたのアドバイスを追加してください。
            - もとのソースコード部分は変更しないでください。不備や改善余地がある場合は別にアドバイスとしてソースコード外でコメントしてください。
            # 入力
            -これから添付するソースコード
            # 出力
            -コピペしてエディタに貼り付けられるようにソースコードの形
    """,
    "エラー調査": "CCC",
}
initial_system_content = ROLE_MAP.get(role_name)
# サイドバーで選択したsystemロールを先頭に常に反映
if "messages" in st.session_state and st.session_state["messages"]:
    st.session_state["messages"][0]["content"] = initial_system_content
# セッションステートに会話履歴を初期化
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": initial_system_content}
    ]

st.title("💬 ChatGPT 風チャット")
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("メッセージを入力...", key="user_input")
    send = st.form_submit_button("送信")
clear = st.button("🗑️ クリア", use_container_width=True)

# クリアボタン押下時は履歴を初期化してリロード
if clear:
    st.session_state["messages"] = [
        {"role": "system", "content": initial_system_content}
    ]
    st.rerun()

# メッセージ送信処理
if send and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})

    assistant_response = ""
    with st.spinner("送信中…"):
        try:
            kwargs = {
                "model": model_name,
                "messages": st.session_state["messages"],
                "stream": True,
            }
            if model_name not in ["o3", "o3-mini"]:
                kwargs["temperature"] = temperature
            stream = client.chat.completions.create(**kwargs)
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
import streamlit.components.v1 as components
import re

for i, msg in enumerate(st.session_state["messages"][1:]):
    text_to_copy = re.sub(
        r"^(user|assistant):\s*", "", msg["content"], flags=re.IGNORECASE
    )
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        components.html(
            f"""
            <div style="position: relative; margin-top: 5px; padding-top: 0;">
<button id="copy_button_{i}" onclick="copy_{i}()" style="position: absolute; top: -5px; right: 0; margin: 0; padding: 5px; background-color: transparent; border: none; color: #cccccc; cursor: pointer;">この会話をコピー</button>
                <textarea id="copy_text_{i}" style="opacity: 0; position: absolute;">{text_to_copy}</textarea>
                <script>
                    function copy_{i}() {{
                        var btn = document.getElementById("copy_button_{i}");
                        if (btn.timeoutId) {{
                            clearTimeout(btn.timeoutId);
                        }}
                        var copyText = document.getElementById("copy_text_{i}");
                        copyText.select();
                        document.execCommand("copy");
                        btn.innerText = "コピーしました！";
                        btn.timeoutId = setTimeout(function() {{
                            btn.innerText = "この会話をコピー";
                            btn.timeoutId = null;
                        }}, 2000);
                    }}
                </script>
            </div>
            """,
            height=60,
        )
