/*
 * 商家控制台的真浏览器冒烟测试（由 test_web_browser.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌
 * {merchant, manager, reviewer, agent, integration}；RODIT_SMOKE_PERIOD 有推荐的周期（测试准备时
 * 用模式 C 接口提问生成）；RODIT_SMOKE_DEFAULT_PERIOD 服务端默认的复核周期（上一个完整周）。
 * 退出码：0 通过；77 没有 Playwright / 浏览器起不来（Python 端跳过）；其他为失败。
 * 标准输出最后一行是 JSON 结果（steps 为走过的步骤）。数据全部是合成的演示数据。
 */
'use strict';

const fs = require('fs');

const SKIP = 77;
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch (err) {
  process.stderr.write(`没有 playwright 模块：${err.message}\n`);
  process.exit(SKIP);
}

const BASE = process.env.RODIT_BASE_URL;
const TOKENS = JSON.parse(process.env.RODIT_SMOKE_TOKENS || '{}');
const PERIOD = process.env.RODIT_SMOKE_PERIOD;
const DEFAULT_PERIOD = process.env.RODIT_SMOKE_DEFAULT_PERIOD;
const FALLBACK_CHROMIUM = '/opt/pw-browsers/chromium';

const steps = [];

function check(condition, message) {
  if (!condition) throw new Error(`检查失败：${message}`);
}

async function launch() {
  try {
    return await chromium.launch();
  } catch (err) {
    if (fs.existsSync(FALLBACK_CHROMIUM)) {
      try {
        return await chromium.launch({ executablePath: FALLBACK_CHROMIUM });
      } catch (err2) {
        process.stderr.write(`Chromium 起不来：${err2.message}\n`);
      }
    }
    process.stderr.write(`Chromium 起不来：${err.message}\n`);
    process.exit(SKIP);
  }
  return null;
}

async function main() {
  const browser = await launch();
  const problems = [];
  const urls = [];
  try {
    // 手机宽度：表格、页签都不能把页面撑出横向滚动
    const context = await browser.newContext({ viewport: { width: 375, height: 800 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      // 故意触发的 401 / 403 / 409 / 503 会有 "Failed to load resource"，不算问题；CSP 拦截、脚本报错算
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    page.on('request', (req) => urls.push(req.url()));

    const tab = (name) => page.getByRole('tab', { name, exact: true });
    const panel = (id) => page.locator(`#panel-${id}`);
    const noHorizontalScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
    async function login(token) {
      await page.getByLabel('访问令牌').fill(token);
      await page.getByRole('button', { name: '登录', exact: true }).click();
    }
    async function signOut() {
      await page.getByRole('button', { name: '退出' }).click();
      await page.getByLabel('访问令牌').waitFor();
      check((await page.evaluate(() => sessionStorage.length)) === 0, '退出清掉 sessionStorage');
      check(await page.locator('#app').isHidden(), '退出后隐藏控制台');
      check((await page.locator('#faqs').textContent()) === '', '退出后不留上一位的数据');
    }

    // ---------------------------------------------------------------- 页面、错误令牌、平台对接、客服
    await page.goto(`${BASE}/console`);
    await page.getByRole('heading', { name: '登录' }).waitFor();
    check((await page.locator('#platform-name').textContent()) === '淘宝', '平台名');
    const links = await page.locator('a.page-link').evaluateAll((as) => as.map((a) => a.getAttribute('href')));
    check(JSON.stringify(links) === JSON.stringify(['assist', 'aftersale']), '页头链接到话术助手和售后审核工作台');
    check(await page.locator('#aftersale-link').isVisible(), '登录前显示去售后审核的链接');
    check(await page.locator('#app').isHidden(), '登录前不显示控制台');
    await login('wrong-token-0000000000000000');
    await page.locator('#error').filter({ hasText: '401' }).waitFor();
    await login(TOKENS.integration);
    await page.locator('#error').filter({ hasText: '平台对接' }).waitFor();
    await login(TOKENS.agent);
    await page.locator('#error').filter({ hasText: '话术助手' }).waitFor();
    check((await page.locator('#error a').getAttribute('href')) === 'assist', '客服：给出去话术助手的链接');
    check(await page.locator('#app').isHidden(), '客服看不到控制台');
    // 令牌留给话术助手，但页头显示是谁、可以退出（不然看起来没登录）
    check((await page.locator('#identity').textContent()).includes('客服'), '客服：页头显示身份');
    check(await page.getByRole('button', { name: '退出' }).isVisible(), '客服：有退出按钮');
    check(await page.locator('#aftersale-link').isVisible(), '客服：页头有去售后审核的链接');
    steps.push('login-guards');

    // ---------------------------------------------------------------- 店长：只读
    await login(TOKENS.manager);
    await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
    check(await page.locator('#error').isHidden(), '登录后清掉错误');
    for (const name of ['知识', '政策', '商品', '设置', '上线', '报表']) check(await tab(name).isVisible(), `店长看得到「${name}」`);
    check(await tab('复核').isHidden(), '店长看不到「复核」');
    check((await tab('知识').getAttribute('aria-selected')) === 'true', '默认第一个页签');
    await panel('knowledge').locator('#faqs .faq').first().waitFor();
    check(await panel('knowledge').locator('.readonly-note').isVisible(), '只读提示');
    check(await page.locator('#faq-create').isHidden(), '店长不能新增 FAQ');
    check((await panel('knowledge').getByRole('button', { name: '修改' }).count()) === 0, '店长没有修改按钮');
    // 键盘：方向键切页签
    await tab('知识').focus();
    await page.keyboard.press('ArrowRight');
    check((await tab('政策').getAttribute('aria-selected')) === 'true', '方向键切到下一个页签');
    await panel('policies').locator('.policy').first().waitFor();
    check((await panel('policies').getByRole('button', { name: /修改|填写/ }).count()) === 0, '店长不能改政策');
    await tab('上线').click();
    await page.locator('#stage-controls').filter({ hasText: '切换阶段由商家' }).waitFor();
    check((await page.locator('#stage-controls button').count()) === 0, '店长没有切换阶段的按钮');
    await signOut();
    steps.push('manager');

    // ---------------------------------------------------------------- 商家：知识（新增、提醒、修改、审核）
    await login(TOKENS.merchant);
    await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
    await tab('知识').click();
    const knowledge = panel('knowledge');
    await knowledge.locator('#candidates .faq').filter({ hasText: '可以寄到新疆吗' }).waitFor();
    await page.locator('#faq-create summary').click();
    const evil = '<img src=x onerror="window.__xss=1">发票抬头可以写公司吗';
    await page.getByLabel('买家问题').fill(evil);
    await page.getByLabel('标准回答').fill('亲，可以开公司抬头的电子普通发票哦~');
    await page.getByLabel('问题类型').selectOption('invoice_gift');
    await page.getByRole('button', { name: '保存并上线' }).click();
    await page.locator('#faq-result').filter({ hasText: '已保存并上线' }).waitFor();
    const created = knowledge.locator('#faqs .faq').filter({ hasText: '发票抬头可以写公司吗' });
    await created.waitFor();
    check((await created.locator('.faq-title').textContent()).includes(evil), 'FAQ 问题原样显示');
    check((await page.locator('#panel-knowledge img').count()) === 0, '没有生成标签');
    check((await page.evaluate(() => window.__xss)) === undefined, '没有执行');
    check((await page.getByLabel('买家问题').inputValue()) === '', '保存后清空表单');

    await page.getByLabel('买家问题').fill('可以加微信聊吗');
    await page.getByLabel('标准回答').fill('亲，加微信可以绝对最低价哦~');
    await page.getByRole('button', { name: '保存并上线' }).click();
    const warnings = page.locator('#faq-result .warnings.blocking').filter({ hasText: '微信' });
    await warnings.waitFor();
    check((await warnings.textContent()).includes('回答时用不上'), '会被拦截的提醒醒目显示');
    check((await warnings.locator('li.blocks').count()) >= 2, '逐条列出提醒');

    const faq1 = knowledge.locator('#faqs .faq').filter({ hasText: '发什么快递' });
    await faq1.getByRole('button', { name: '修改' }).click();
    await faq1.getByLabel('标准回答').fill('亲，默认发中通快递，顺丰补 10 元哦~');
    await faq1.getByRole('button', { name: '保存修改' }).click();
    await page.locator('#knowledge-result').filter({ hasText: '旧条目 #1 已下线' }).waitFor();
    check((await page.locator('#knowledge-result').textContent()).includes('新条目 #'), '修改：显示新条目编号');

    const candidate = knowledge.locator('#candidates .faq').filter({ hasText: '可以寄到新疆吗' });
    await candidate.getByRole('button', { name: '修改后通过' }).click();
    await candidate.getByRole('button', { name: '按修改后的内容通过' }).click();
    await candidate.locator('.status.warn').filter({ hasText: '没有改动' }).waitFor();
    await candidate.getByLabel('标准回答').fill('亲，新疆可以发，补 15 元运费哦~');
    await candidate.getByRole('button', { name: '按修改后的内容通过' }).click();
    await page.locator('#knowledge-result').filter({ hasText: '候选 #6 已下线' }).waitFor();
    await knowledge.locator('#candidates').filter({ hasText: '没有待审核的候选知识' }).waitFor();
    check((await page.locator('#faq-status').inputValue()) === 'approved', 'FAQ 列表默认只看已上线');
    await page.locator('#faq-status').selectOption('retired');
    await knowledge.locator('#faqs .faq.st-retired').filter({ hasText: '发什么快递' }).waitFor();
    check((await knowledge.locator('#faqs .faq.st-approved').count()) === 0, '按状态筛选');
    await page.locator('#faq-status').selectOption('approved');
    steps.push('knowledge');

    // ---------------------------------------------------------------- 商家：政策
    await tab('政策').click();
    const warranty = panel('policies').locator('.policy').filter({ has: page.getByRole('heading', { name: '保修' }) });
    await warranty.locator('.policy-value.empty').filter({ hasText: '未填写' }).waitFor();
    await warranty.getByRole('button', { name: '填写' }).click();
    await warranty.getByLabel('保修', { exact: false }).fill('保修一年，联系客服寄回维修');
    await warranty.getByRole('button', { name: '保存' }).click();
    const saved = panel('policies').locator('.policy').filter({ hasText: '保修一年' });
    await saved.locator('.status.ok').filter({ hasText: '已保存「保修」' }).waitFor();
    // 政策不再适用：清空（要确认），之后显示"未填写"
    await saved.getByRole('button', { name: '清空' }).click();
    await saved.getByRole('button', { name: '确认清空' }).click();
    const cleared = panel('policies').locator('.policy').filter({ has: page.getByRole('heading', { name: '保修' }) });
    await cleared.locator('.status.ok').filter({ hasText: '已清空「保修」' }).waitFor();
    await cleared.locator('.policy-value.empty').filter({ hasText: '未填写' }).waitFor();
    steps.push('policies');

    // ---------------------------------------------------------------- 商家：商品卡片、同步
    await tab('商品').click();
    const row777 = page.locator('#items tr', { hasText: '高腰阔腿牛仔裤' });
    await row777.waitFor();
    check((await row777.locator('.chip.missing').filter({ hasText: '尺码建议' }).count()) === 1, '列表显示没填的字段');
    await row777.getByRole('button', { name: '查看 / 编辑' }).click();
    const detail = page.locator('#item-detail');
    await detail.getByRole('heading', { name: /高腰阔腿牛仔裤/ }).waitFor();
    check((await detail.locator('.item-platform').textContent()).includes('平台字段'), '平台字段只读');
    await detail.getByLabel('尺码建议').fill('版型偏大，建议拍小一码');
    await detail.getByLabel('洗涤保养').fill('建议手洗\n不可暴晒\n建议手洗');
    await detail.getByRole('button', { name: '保存商家填写的字段' }).click();
    await detail.locator('.status.ok').filter({ hasText: '已保存：尺码建议' }).waitFor();
    await page.locator('#items tr', { hasText: '高腰阔腿牛仔裤' }).locator('.chip.filled').filter({ hasText: '尺码建议' }).waitFor();
    check((await detail.getByLabel('洗涤保养').inputValue()) === '建议手洗\n不可暴晒', '列表去重后显示');

    await page.getByRole('button', { name: '同步商品' }).click();
    const report = page.locator('#sync-report');
    await report.filter({ hasText: '同步完成（全店）' }).waitFor();
    check((await report.textContent()).includes('1 件：i_999'), '同步报告列出新建的商品');
    await page.locator('#items tr', { hasText: 'i_999' }).waitFor();
    await page.route('**/console/items/sync', (route) =>
      route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'sync_unavailable', message: '这个单元没有接入平台商品接口' }),
      }),
    );
    await page.getByRole('button', { name: '同步商品' }).click();
    await report.filter({ hasText: '平台商品同步未接入' }).waitFor();
    check(await page.locator('#error').filter({ hasText: '平台商品同步未接入' }).isVisible(), '503 也进错误区');
    await page.unroute('**/console/items/sync');
    check(await noHorizontalScroll(), '商品页在手机宽度下没有横向滚动');
    steps.push('items');

    // ---------------------------------------------------------------- 商家：店铺设置
    await tab('设置').click();
    const tone = page.locator('.config-row[data-key="tone"]');
    await tone.getByLabel('修改语气').selectOption('专业');
    await tone.getByRole('button', { name: '保存' }).click();
    await page.locator('.config-row[data-key="tone"] .status.ok').filter({ hasText: '已保存：语气 → 专业' }).waitFor();
    const count = page.locator('.config-row[data-key="suggestion_count"]');
    await count.getByRole('button', { name: '恢复默认' }).click();
    await page.locator('.config-row[data-key="suggestion_count"] .status.ok').filter({ hasText: '已恢复默认' }).waitFor();
    check(
      (await page.locator('.config-row[data-key="suggestion_count"] .badge').textContent()) === '默认',
      '恢复默认后显示来源',
    );
    const maxChars = page.locator('.config-row[data-key="max_answer_chars"]');
    await maxChars.getByLabel('修改回答最多字数').fill('9999');
    await maxChars.getByRole('button', { name: '保存' }).click();
    await maxChars.locator('.status.error').filter({ hasText: '20–500' }).waitFor();
    const disabled = page.locator('.config-row[data-key="disabled_intents"]');
    await disabled.getByLabel('问候 / 催回复').check();
    await disabled.getByRole('button', { name: '保存' }).click();
    await page.locator('.config-row[data-key="disabled_intents"] .status.ok').filter({ hasText: '问候 / 催回复' }).waitFor();
    check(
      (await page.locator('.config-row[data-key="shadow_mode"]').textContent()).includes('不能在这里修改'),
      '影子模式开关只读并说明原因',
    );
    check(await noHorizontalScroll(), '设置页在手机宽度下没有横向滚动');
    steps.push('config');

    // ---------------------------------------------------------------- 商家：上线（只能切回影子）、报表
    await tab('上线').click();
    await page.locator('#stage-label').filter({ hasText: '智能辅助' }).waitFor();
    check((await page.getByRole('button', { name: '切到智能辅助' }).count()) === 0, '商家不能切到智能辅助');
    await page.getByRole('button', { name: '切回影子模式' }).click();
    await page.getByRole('button', { name: '确认切回影子模式' }).click();
    await page.locator('#stage-controls .status.ok').filter({ hasText: '智能辅助 → 影子模式' }).waitFor();
    await page.locator('#stage-label').filter({ hasText: '影子模式' }).waitFor();
    await tab('报表').click();
    await page.getByLabel('统计范围').selectOption('');
    await panel('reports').getByRole('button', { name: '刷新' }).click();
    await page.locator('#adoption .headline').waitFor();
    await page.locator('#qa-report dl').waitFor();
    check(await noHorizontalScroll(), '报表在手机宽度下没有横向滚动');
    await signOut();
    steps.push('merchant');

    // ---------------------------------------------------------------- 复核人员：抽样、打分、报表
    await login(TOKENS.reviewer);
    await page.locator('#identity').filter({ hasText: '复核人员' }).waitFor();
    check((await tab('复核').getAttribute('aria-selected')) === 'true', '复核人员默认进「复核」');
    check(await page.locator('#aftersale-link').isHidden(), '复核人员：页头不显示去售后审核的链接');
    for (const name of ['知识', '政策', '商品', '设置', '报表']) check(await tab(name).isHidden(), `复核人员看不到「${name}」`);
    check(await page.locator('.access-notice').filter({ hasText: '访问日志' }).isVisible(), '访问留痕提示');
    await page.locator('#review-report .summary').filter({ hasText: DEFAULT_PERIOD }).waitFor();
    check((await page.getByLabel('周期（ISO 周）').inputValue()) === DEFAULT_PERIOD, '默认上一个完整周');
    await page.getByLabel('周期（ISO 周）').fill(PERIOD);
    await page.getByRole('button', { name: '抽样' }).click();
    await page.locator('#review-status.ok').filter({ hasText: '本次新抽' }).waitFor();
    const items = page.locator('#review-queue .review-item');
    await items.first().waitFor();
    const total = await items.count();
    check(total >= 2, '抽中了样本');
    await page.locator('#review-progress-text').filter({ hasText: `已复核 0 / ${total}` }).waitFor();
    const first = items.first();
    check((await first.locator('.card-text').first().textContent()).length > 0, '显示推荐原文');
    check((await first.locator('blockquote.buyer').textContent()).length > 0, '显示买家问题');
    await first.getByRole('button', { name: '提交评分' }).click();
    await first.locator('.status.error').filter({ hasText: '事实正确' }).waitFor();
    for (const choice of ['2 分 · 完全正确', '2 分 · 完整回答', '2 分 · 无问题', '1 分 · 可以接受']) {
      await first.getByLabel(choice).check();
    }
    await first.getByLabel('备注（选填，保存前会打码）').fill('表达略生硬');
    await first.getByRole('button', { name: '提交评分' }).click();
    const scored = page.locator('#review-queue .review-item').first();
    await scored.locator('.status.ok').filter({ hasText: '已记录样本' }).waitFor();
    check((await scored.getByRole('button', { name: '修改评分' }).count()) === 1, '打过分的可以改判');
    await page.locator('#review-progress-text').filter({ hasText: `已复核 1 / ${total}` }).waitFor();
    const second = page.locator('#review-queue .review-item').nth(1);
    await second.getByRole('button', { name: '跳过' }).click();
    await second.locator('.status.ok').filter({ hasText: '已跳过' }).waitFor();
    await page.locator('#review-progress-text').filter({ hasText: `已复核 2 / ${total}` }).waitFor();
    const factualRow = page.locator('#review-report tr', { hasText: '事实正确' }).first();
    check((await factualRow.locator('td').first().textContent()) === '1', '报表：已打分 1 条');
    check((await page.locator('#quality-gate-status').textContent()) === '数据不足', '§13.2 门槛：样本不够');
    steps.push('review');

    // ---------------------------------------------------------------- 复核人员：切到智能辅助（门槛没过 → 强制）
    await tab('上线').click();
    await page.locator('#stage-label').filter({ hasText: '影子模式' }).waitFor();
    await page.getByRole('button', { name: '切到智能辅助' }).click();
    const failure = page.locator('.gate-failure');
    await failure.filter({ hasText: '门槛没过' }).waitFor();
    check((await failure.locator('tbody tr').count()) >= 3, '显示 409 带回的门槛结果');
    await failure.getByRole('button', { name: '强制切到智能辅助' }).click();
    await page.locator('#stage-controls .status.error').filter({ hasText: '写明原因' }).waitFor();
    await failure.getByLabel('强制切换的原因', { exact: false }).fill('测试集已人工确认达标');
    await failure.getByRole('button', { name: '强制切到智能辅助' }).click();
    await page.locator('#stage-controls .status.ok').filter({ hasText: '强制切换' }).waitFor();
    await page.locator('#stage-label').filter({ hasText: '智能辅助' }).waitFor();
    check(await page.getByRole('button', { name: '切到影子模式' }).isVisible(), '复核人员可以切回影子');
    check(await noHorizontalScroll(), '上线页在手机宽度下没有横向滚动');
    steps.push('stage');

    // ---------------------------------------------------------------- 复核人员打开话术助手：提示去控制台，点链接回来
    await page.goto(`${BASE}/assist`);
    await page.locator('#error').filter({ hasText: '复核人员请使用商家控制台' }).waitFor();
    check(await page.locator('#assist').isHidden(), '话术助手页不给复核人员显示工具');
    check(await page.getByRole('button', { name: '退出' }).isVisible(), '复核人员：话术助手页有退出按钮');
    check(await page.locator('#aftersale-link').isHidden(), '复核人员：话术助手页也不显示去售后审核的链接');
    await page.locator('#error a').click();
    await page.locator('#identity').filter({ hasText: '复核人员' }).waitFor();
    check(page.url() === `${BASE}/console`, '回到控制台');
    await signOut();
    check(await page.locator('#aftersale-link').isVisible(), '退出后照常显示去售后审核的链接');
    steps.push('assist-link');

    // ---------------------------------------------------------------- 令牌从不出现在网址里
    const secrets = Object.values(TOKENS);
    check(!urls.some((u) => secrets.some((t) => u.includes(t))), '令牌不进网址');
    check(problems.length === 0, `浏览器报错：${problems.join(' | ')}`);
    process.stdout.write(`${JSON.stringify({ ok: true, steps, requests: urls.length, samples: total })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stdout.write(`${JSON.stringify({ ok: false, steps, error: String(err && err.stack ? err.stack : err) })}\n`);
  process.exit(1);
});
