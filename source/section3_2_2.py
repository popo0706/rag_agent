# ================================================================
# 【概要】
# OpenAI Chat Completion API を使い、
# 「回答を100文字程度で返してね」と依頼する最小サンプルです。
# 1. OpenAI クライアントを生成（API キーは環境変数から自動取得）
# 2. system メッセージで「100文字程度」と指示
# 3. `max_tokens` を 70 に絞って文字数オーバーを抑制
# 4. 返答を取り出し `print()` で表示
# ================================================================

from openai import OpenAI  # OpenAI クライアントクラスの読み込み

client = OpenAI()  # API キーは環境変数 OPENAI_API_KEY から自動検出

response = client.chat.completions.create(
    model="gpt-4.1-nano",  # 軽量・高速な GPT-4.1 系モデル
    # ▼ 会話履歴（messages）をリストで渡す
    messages=[
        {
            "role": "system",
            "content": "回答は100文字程度にしてください。",
            # ↑ “system” で AI へのルールを宣言。
            #    ここでは「文字数制限」という執筆ガイドラインを伝える。
        },
        {
            "role": "user",
            "content": "プロンプトエンジニアリングとファインチューニングの違いは？",
            # ↑ ユーザーの質問。100文字以内で答えてほしいお題。
        },
    ],
    # ▼ 念のため出力トークン上限を 70 に設定
    #    - 70 トークン ≒ 全角 100 文字弱に相当（厳密ではない）
    #    - 文字数制限をシステムプロンプトで指示しつつ、
    #      max_tokens でも二重に制限しておくとオーバーしにくい。
    max_tokens=70,
)

# ▼ ChatGPT の返答（assistant メッセージ）を取り出して表示
print(response.choices[0].message.content)
#    response.choices は「複数案が返る」可能性があるリスト。
#    通常は index 0 が最も確度の高い候補。
