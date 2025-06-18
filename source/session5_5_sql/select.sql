-- SQLite
SELECT
  id,                                -- 行番号
  json_extract(message, '$.type')           AS role,    -- human / ai
  json_extract(message, '$.data.content')   AS content  -- 日本語テキスト
  ,*
FROM message_store

