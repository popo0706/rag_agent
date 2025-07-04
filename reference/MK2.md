1. 全体像をつかむ３ステップ
ステップ	やること	ゴール
① ヒアリング準備	質問リストで“抜け”をふさぐ	現場の数字（KPI）と制約条件を集める
② PoC（厨房で味見）	超小規模で技術検証	「動く／効果あり」を早期確認
③ 実証実験→本番	現場でリハーサル→正式導入	ビジネス成果と運用フローを固める
2. ヒアリングシート（穴埋め式）
以下を「★仮説」で先に書き込み、打合せで○×をもらうと話が早いです。

A. ビジネス／KPI
質問	回答例（仮）
チェック1件の手作業時間は？	★60分
見逃しミスは月何件？	★3件
どこまで下げれば“成功”か？(KPI)	★時間50％↓、ミス率1件未満
B. データ
質問	回答例（仮）
計画書ファイル形式	★PDF 80％、Word 20％
平均ページ数/容量	★30p / 5MB
言語	★日本語のみ
過去データ件数	★200件（S3に保管）
C. チェックリスト
質問	回答例（仮）
パターン数	★3種類（開発/保守/研究）
項目タイプ	★Yes/No 80％、自由記述20％
改訂頻度	★年4回
D. システム利用
質問	回答例（仮）
同時ユーザ数	★10
応答時間目標	★5秒以内
AWS制約	★Bedrock使用OK、リージョン=ap-northeast-1
3. 技術ベースライン（RAG on AWS）
[ユーザ]─HTTPS─> API Gateway
                 │
                 ▼
          AWS Lambda (Prompt & Glue)
                 │
  ┌───────────┬────────────┐
  │LLM (Claude, Llama3) │OpenSearch Serverless│
  │ via Bedrock         │  (ベクトルDB)        │
  └───────────┴────────────┘
                 │
                 ▼
               S3 (文書原本・ログ)

ポイント

99％マネージドで“作ってすぐ動く”
Embedding は Bedrock Titan or Cohere multilingual
PoC は Serverless、本番で規模拡大なら OpenSearch Cluster / Aurora + pgvector へ差し替え
4. PoC と実証実験の違い
項目	PoC（Proof of Concept）	実証実験（パイロット）
目的	技術的に「作れる？」	現場で「役立つ？」
データ	5～10件	数百件＋リアル運用
関係者	開発チーム中心	現場ユーザ・情シス・監査
期間	1～3週	1～3か月
失敗時	学びになればOK	影響大なのでリスク計画必須
5. 進め方タイムライン（例）
0週　キックオフ準備
★上のヒアリングシートを仮入力し共有
1～2週　要件ヒアリング＋PoC設計
3～5週　PoC実装・レビュー
評価尺度＝KPI（時間短縮・ミス率）
6週　Go/NoGo判定
7～10週　要件定義＋本番アーキテクチャ確定
11～18週　開発・テスト
19週～　実証実験（パイロット）
25週～　本番リリース
6. 想定される悩みと対策
悩み	原因	早期対策
① ハルシネーション	検索ヒットが薄い	類似度閾値↑・回答拒否を許容
② コスト爆発	LLMトークン量	Chunk小さく・Bedrock Comprehend割引枠
③ チェックリスト改訂多発	手作業で修正	YAML外部化＋Git管理
④ 精度が上がらない	教師データ不足	PoCでラベル付けフロー確立
⑤ 権限・ログ監査	監査部門要件	CloudTrail＋S3・Glacier保管
7. すぐ着手すべき３アクション
ヒアリングシートを顧客アジェンダとして配布
**サンプル文書＆チェックリスト（数件）**を借用する手続きを開始
社内でBedrock利用の可否と概算予算を先に確認
8. 追加で確認したいこと（質問テンプレ）
### 業務面
- 品質管理プロセスの月次件数は？
- 手作業チェックの標準手順書はある？

### セキュリティ
- S3暗号化キーは顧客管理かAWS管理か？
- 個人情報は含まれるか？

### 運用
- モデル／データ更新の担当部署と頻度は？
- 障害対応（24/365？ 平日9-17？）

### 予算・契約
- PoCフェーズの上限コストは？
- 本番後の運用費は年間いくらを想定？

9. 用語ミニ辞典
KPI：途中経過を測る数字のものさし
KGI：最終ゴールのゴールテープ
PoC：「作れるか？」を試す小さな実験
実証実験：「役立つか？」を試す現場リハーサル
RAG：検索＋生成で答えるAIの王道構成
以上が「要件定義～技術アドバイスの進め方」「事前確認ポイント」「想定リスクと対応策」をまとめた最新版です。
不足や疑問があればいつでもお知らせください！