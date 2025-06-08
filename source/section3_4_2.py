# ===============================================================
# 【要約】
# Few-shot プロンプティングの最小例です。
# 「AI に関係する文章なら true、そうでないなら false」と
# 分類させるルールを、たった 2 組の例示（Few-shot）だけで
# GPT-4.1-nano に学習させ、新しい入力文を判定します。
# ===============================================================

# --------------------------------------------
# Few-shot プロンプティングの概要
# --------------------------------------------
# 「Few-shot」は「少しのお手本を見せる」という意味。
# つまり、いくつか例（＝shot）を提示したうえで
# “次の入力も同じルールで答えてね” と AI に教える方法です。
# --------------------------------------------

from openai import OpenAI  # ← OpenAI SDK のクライアントクラスをインポート

client = OpenAI()  # ← 環境変数 OPENAI_API_KEY を読み取り、クライアントを生成

response = client.chat.completions.create(  # ← Chat Completions API を呼び出す
    model="gpt-4.1-nano",  # 使うモデル。GPT-4.1 ベースの軽量＆高速版
    messages=[
        # ① system メッセージ ── モデル全体への指示
        {
            "role": "system",
            "content": "入力がAIに関係するか回答してください。",
            # ↑ ここで “AI 関連なら true / そうでなければ false” という
            #    シンプルなルールを宣言している。
        },
        # ②〜⑥ Few-shot の「お手本」ペアを定義
        #    例を複数提示して “AI 関連なら true, そうでなければ false”
        #    という分類ルールを学習させる
        {"role": "user", "content": "AIの進化はすごい。"},  # ← AI に関する文
        {"role": "assistant", "content": "true"},  # ← 正解ラベル
        {"role": "user", "content": "今日の天気は良い。"},  # ← AI と無関係な文
        {"role": "assistant", "content": "false"},  # ← 正解ラベル
        # ⑦ 最後の user メッセージ ── これが実際に分類したい新しい入力
        {
            "role": "user",
            "content": "ChatGPTは便利だ。",  # ← AI に関する文なので true が期待値
        },
    ],
)

# ▼ ChatGPT が返す分類結果（"true" / "false" のどちらか）は
#    response.choices[0].message.content に格納される。
print(response.choices[0].message.content)
