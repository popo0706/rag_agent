# ChatGPT 風チャットアプリ

このアプリは Python と Streamlit による ChatGPT 風チャット UI です。  
OpenAI の Chat Completions API を利用して回答を取得します。

## セットアップ手順

1. Python 仮想環境の作成と有効化  
   Windows の場合:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   macOS/Linux の場合:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. 依存ライブラリのインストール  
   ```
   pip install -r requirements.txt
   ```

3. 環境変数の設定  
   ```
   cp .env.example .env
   ```
   または secrets.toml を利用する場合は適宜設定してください。

4. アプリの起動  
   ```
   streamlit run ./chatgpt_like_app/app.py
   ```

5. ブラウザで http://localhost:8501 へアクセス

## 注意事項

- API キーは .env または secrets.toml に設定してください。
- 会話履歴はセッション中のみ保持され、DB 等は使用していません。
