# pipコマンド便利集
## インストール済みパッケージ一覧
pip list
## 矛盾チェック(バージョン衝突や欠損依存 があるとリストアップされます)
pip check
## スナップショットを残す（バックアップ）
pip freeze > requirements_backup.txt
pip freeze > rags_requirements_backup.txt
### バックアップからの復旧
pip install -r requirements_backup.txt