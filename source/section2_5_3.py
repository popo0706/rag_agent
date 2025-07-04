# ================================================================
# 【概要】
# OpenAI 公式ライブラリ tiktoken を使って、与えた文章を
# “モデルが実際に読む単位＝トークン” に分割し、その個数を数える
# 最小サンプルです。料金試算や入力長チェックに役立ちます。
# ================================================================


# ──────────────────────────────────────────────────────────────
# 文章を「トークン」に分割して、その個数を数えるサンプル
# ──────────────────────────────────────────────────────────────
#  ❑ 「トークン」とは？
#       - モデルが実際に読む“かたまり”の単位（単語より細かい）。
#       - 日本語だと 1 文字ずつ切れたり、英語だと接頭・接尾で切れたりする。
#       - モデルごとに“分け方（エンコードルール）”が違うので注意！
#
#  ❑ なぜ数えるの？
#       - OpenAI API は「トークン数 × 単価」で料金が決まる。
#       - 入力上限（例: gpt-4o は 128k）を超えないか事前に確認できる。
# ──────────────────────────────────────────────────────────────

from openai import OpenAI  # v1 系で追加されたクライアントクラス
import tiktoken  # OpenAI 公式トークン分割ライブラリ

client = OpenAI()  # OpenAI API の窓口を生成
# （API キーは環境変数 OPENAI_API_KEY から自動読込）


# ※ 事前に `pip install tiktoken` が必要

# 数えたい文章を用意（日本語＋句読点入り）
text = (
    "LLMを使ってクールなものを作るのは簡単だが、"
    "プロダクションで使えるものを作るのは非常に難しい。"
)
#  ↑ 括弧内で改行しても Python は自動で連結。長文を可読フォーマットで書ける。

# ① モデルに合った“エンコーディングルール”を取得
#    encoding_for_model() にモデル名を渡すと、対応する分割方法を返してくれる。
#    GPT-3.5 系と GPT-4 系ではルールが微妙に異なるので毎回これを呼ぶのが安全。
encoding = tiktoken.encoding_for_model("gpt-4o")

# ② 文章をトークンIDのリストへエンコード
#    encode() の戻り値は整数 ID の配列例: [10123, 20987, ...]
tokens = encoding.encode(text)

# ③ トークン配列の長さ = トークン数を len() で取得
token_count = len(tokens)

# ④ 結果を表示
print(token_count)  # 例: 24 と表示される
#  *ポイント*
#    - 文章を変えれば当然数も変わる。長いほどトークン数は増加。
#    - 料金や入力上限を見積もるときは「入力＋出力の合計」で考える！
