-- Rodit 数据库结构，迁移版本 3：
-- 1. 撤回先于消息到达（§8.3 平台推送乱序、§5.3 撤回的消息不再进上下文）：message_recall 记下
--    本店还没见过、就先被撤回的消息（按店铺 + 会话 + 消息 ID），有效期内这条消息到达时丢弃；
--    用掉（消息到达）或过期即删，由 PgConversationRepository 维护。
-- 2. 等商家审核的候选知识不重复，多个实例之间也一样（§5.8）：knowledge_item 上"同一店铺、同样的
--    问题和回答、同一商品、同一问题类型、状态为 candidate"的部分唯一索引。问题和回答按 md5 进索引
--    （长文本放不进 B 树索引项）；商品、问题类型为空按空串算，两条都为空的也算相同。
--
-- 版本 1、2 已发布，不再修改。与之前的版本一样可以重复执行（IF NOT EXISTS、DROP POLICY IF EXISTS）。

-- ---------------------------------------------------------------- 撤回墓碑

CREATE TABLE IF NOT EXISTS message_recall (
  shop_id         TEXT NOT NULL CHECK (shop_id <> ''),
  conversation_id TEXT NOT NULL,
  msg_id          TEXT NOT NULL,                      -- 被撤回、但还没到达的消息
  recalled_at     TIMESTAMPTZ NOT NULL,
  expires_at      TIMESTAMPTZ NOT NULL,               -- 过期后不再生效，写入墓碑时顺带清理
  PRIMARY KEY (shop_id, conversation_id, msg_id)
);
CREATE INDEX IF NOT EXISTS message_recall_expiry_idx ON message_recall (shop_id, expires_at);

ALTER TABLE message_recall ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_recall FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS shop_isolation ON message_recall;
CREATE POLICY shop_isolation ON message_recall
  USING (shop_id = current_setting('rodit.shop_id', true))
  WITH CHECK (shop_id = current_setting('rodit.shop_id', true));
GRANT SELECT, INSERT, UPDATE, DELETE ON message_recall TO rodit_app;

-- ---------------------------------------------------------------- 候选知识去重

-- 已有的重复候选（唯一索引建不起来）：每组留 id 最小的一条，其余改成已下线（retired）——
-- 不删行，审计里记着的候选 id 仍然查得到。迁移账号是表的所有者（不是超级用户）时也受 FORCE
-- 行级安全约束、没有店铺范围时一行都看不到，所以在本事务里临时取消 FORCE（所有者豁免行级安全），
-- 改完立即恢复；整个迁移在一个事务里，其他会话看不到中间状态。
ALTER TABLE knowledge_item NO FORCE ROW LEVEL SECURITY;
UPDATE knowledge_item k
   SET status = 'retired'
 WHERE k.status = 'candidate'
   AND EXISTS (
     SELECT 1 FROM knowledge_item d
      WHERE d.status = 'candidate'
        AND d.shop_id = k.shop_id
        AND d.id < k.id
        AND d.question = k.question
        AND d.answer = k.answer
        AND d.item_id IS NOT DISTINCT FROM k.item_id
        AND d.intent IS NOT DISTINCT FROM k.intent
   );
ALTER TABLE knowledge_item FORCE ROW LEVEL SECURITY;

CREATE UNIQUE INDEX IF NOT EXISTS knowledge_item_pending_candidate_uniq
  ON knowledge_item (
    shop_id, md5(question), md5(answer), COALESCE(item_id, ''), COALESCE(intent, '')
  )
  WHERE status = 'candidate';
