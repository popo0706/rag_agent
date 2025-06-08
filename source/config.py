"""
【概要】
このモジュールは「.env」ファイルに書かれた OpenAI の API キーを読み込み、
`openai` ライブラリにセットするだけのシンプルな設定ファイルです。
たとえばファイル名を config.py として保存しておくと、
他のスクリプト側で

    from config import openai

とインポートするだけで、すでに API キーが設定済みの `openai` オブジェクトを
そのまま利用できます。煩雑なキー設定を各所で繰り返さずに済むため、
コードをきれいに保てます。
"""

from dotenv import load_dotenv  # .env ファイルを読み込むヘルパーライブラリ
import os, openai  # os: 環境変数を扱う標準ライブラリ

# openai: OpenAI API を呼び出す公式ライブラリ

load_dotenv()  # .env に書かれた内容を環境変数へ取り込む
openai.api_key = os.getenv(
    "OPENAI_API_KEY"
)  # 取り込んだ環境変数から API キーを取得して設定
