# ============================================================
# 【概要】
# このスクリプトは 「OpenAI Function Calling」 の最小構成デモです。
# 1. Python 側で get_current_weather() という関数を用意
# 2. その関数の仕様（引数や説明）を tools としてモデルに宣言
# 3. ユーザーが「東京の天気は？」と聞くと、モデルが関数呼び出しを提案
# 4. Python が実際に関数を実行し、結果を会話履歴へ追加
# 5. その履歴を再度モデルに渡し、自然言語の最終回答を取得
# ============================================================

import json  # 標準ライブラリ。JSON文字列⇔Pythonオブジェクトを相互変換できる
from openai import OpenAI  # v1系SDKのクライアントクラス。API呼び出しの窓口になる


# ─────────────────────────────────────────────────────────────
# ① ChatGPT から「実行してほしい」と呼ばれる関数を Python 側で用意
# ─────────────────────────────────────────────────────────────
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """
    指定した都市の “現在の天気” を返す（ここではダミーデータ）。

    Parameters
    ----------
    location : str
        都市名（例: "Tokyo"）
    unit : str, optional
        'celsius' または 'fahrenheit'。既定は 'fahrenheit'。

    Returns
    -------
    dict
        例: {'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'}
    """
    location_lower = location.lower()  # 大文字/小文字の違いを吸収して検索しやすくする

    # 以下は if‐elif で “どの都市か” をざっくり判定
    if "tokyo" in location_lower:  # 「tokyo」が含まれる？
        return {"location": "Tokyo", "temperature": "10", "unit": unit}
    elif "san francisco" in location_lower:  # 「san francisco」が含まれる？
        return {"location": "San Francisco", "temperature": "72", "unit": unit}
    elif "paris" in location_lower:  # 「paris」が含まれる？
        return {"location": "Paris", "temperature": "22", "unit": unit}
    else:  # どれでもなければ「不明値」を返す
        return {"location": location, "temperature": "unknown", "unit": unit}


# ─────────────────────────────────────────────────────────────
# ② ChatGPT 側へ “この関数が使えますよ” と伝えるためのツール定義
# ─────────────────────────────────────────────────────────────
tools = [
    {
        "type": "function",  # 「関数呼び出し用のツールだよ」というおまじない
        "function": {
            "name": "get_current_weather",  # Python の関数名と 1:1 で一致させる
            "description": "Get the current weather in a given location",
            "parameters": {  # 引数の “型宣言” (JSON Schema 風)
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],  # 枚挙型＝取り得る値を限定
                    },
                },
                "required": ["location"],  # location は必須、unit は省略可
            },
        },
    }
]

# ─────────────────────────────────────────────────────────────
# ③ 「東京の天気は？」とユーザーが尋ねるところから会話開始
# ─────────────────────────────────────────────────────────────
client = OpenAI()  # API キーは環境変数 OPENAI_API_KEY から自動検出される

messages = [{"role": "user", "content": "東京の現在の天気は？"}]

response = client.chat.completions.create(
    model="gpt-4.1-nano",  # 軽量モデル（ローカル実行例）
    messages=messages,  # いまの会話履歴
    tools=tools,  # 「使える関数群」を AI に提示
)

# 受信レスポンスを丸ごと確認（開発時のデバッグ用）
print(response.to_json(indent=2))

# ChatGPT が「関数を呼ぶべき」と判断したかどうかをチェック
response_message = response.choices[0].message
messages.append(response_message.to_dict())  # AI からの“関数呼び出し依頼”を履歴へ追加

# ─────────────────────────────────────────────────────────────
# ④ 実際に Python 関数を実行し、その結果を AI へ返す
# ─────────────────────────────────────────────────────────────
available_functions = {"get_current_weather": get_current_weather}  # 名称 → 実体 の辞書

for tool_call in response_message.tool_calls:  # 複数呼び出しに備え for ループ
    function_name = tool_call.function.name  # どの関数を呼びたい？
    function_to_call = available_functions[function_name]  # 実体を取得

    # tool_call.function.arguments は JSON 文字列で渡ってくる → dict へ変換
    function_args = json.loads(tool_call.function.arguments)

    # Python 関数を実行し、辞書形式で結果を得る
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    # 実行結果も会話履歴に積む。role="tool" がポイント。
    messages.append(
        {
            "tool_call_id": tool_call.id,  # 紐づけ用 ID
            "role": "tool",  # 「ツールからの返答」印
            "name": function_name,  # どの関数？
            "content": json.dumps(function_response),  # JSON 文字列で渡す
        }
    )

# ここまでで messages には
#   user → assistant(関数呼び出し依頼) → tool(実行結果)
# の 3 連コンボがそろった
print(json.dumps(messages, ensure_ascii=False, indent=2))

# ─────────────────────────────────────────────────────────────
# ⑤ 関数結果込みの履歴を再度 ChatGPT に渡し、自然言語の最終回答を生成
# ─────────────────────────────────────────────────────────────
second_response = client.chat.completions.create(
    model="gpt-4.1-nano", messages=messages
)

# ChatGPT が人間向けにわかりやすくまとめてくれた返事を確認
print(second_response.to_json(indent=2))
