/*
 * 模式 C 网页工具的真浏览器冒烟测试（由 test_web_browser.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌 {agent, merchant, integration}。
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
    // 手机宽度（§ 页面要能在手机上用）
    const context = await browser.newContext({ viewport: { width: 375, height: 740 }, locale: 'zh-CN' });
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: BASE });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      // 故意触发的 401 / 409 / 503 会有 "Failed to load resource"，不算问题；CSP 拦截、脚本报错算
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    page.on('request', (req) => urls.push(req.url()));

    const question = page.getByLabel('买家的问题');
    const newestTurn = () => page.locator('article.turn').first();
    async function ask(text) {
      await question.fill(text);
      await page.getByRole('button', { name: '获取推荐回答' }).click();
      await newestTurn().locator('.turn-body').filter({ hasNotText: '正在生成推荐' }).waitFor();
      return newestTurn();
    }
    async function login(token) {
      await page.getByLabel('访问令牌').fill(token);
      await page.getByRole('button', { name: '登录' }).click();
    }

    // ---------------------------------------------------------------- 页面与合规提示
    await page.goto(`${BASE}/assist`);
    await page.locator('#notice summary').filter({ hasText: '个人信息委托处理协议' }).waitFor();
    check(await page.getByText('不做任何自动抓取').isVisible(), '登录前展开合规提示');
    check((await page.locator('#platform-name').textContent()) === '淘宝', '平台名');
    steps.push('page');

    // ---------------------------------------------------------------- 登录：错误令牌、平台对接令牌、客服
    await login('wrong-token-0000000000000000');
    await page.locator('#error').filter({ hasText: '401' }).waitFor();
    await login(TOKENS.integration);
    await page.locator('#error').filter({ hasText: '平台对接' }).waitFor();
    check(await page.locator('#assist').isHidden(), '登录失败时不显示工具');
    await login(TOKENS.agent);
    await page.locator('#identity').filter({ hasText: '客服' }).waitFor();
    check(await page.locator('#error').isHidden(), '登录后清掉错误');
    check(await page.locator('#console').isHidden(), '客服看不到店铺数据');
    check(!(await page.locator('#notice').evaluate((d) => d.open)), '登录后收起须知');
    check(await page.getByText('个人信息委托处理协议').isVisible(), '协议要求一直可见');
    await page.locator('#item option', { hasText: '法式碎花连衣裙' }).waitFor({ state: 'attached' });
    check((await page.evaluate(() => sessionStorage.getItem('rodit.assist.token'))) === TOKENS.agent, '令牌在 sessionStorage');
    steps.push('login');

    // ---------------------------------------------------------------- 提问（Ctrl+Enter）→ 复制并标记已发送
    await question.fill('能开发票吗');
    await question.press('Control+Enter');
    const card1 = newestTurn().locator('article.card').first();
    await card1.waitFor();
    const text1 = await card1.locator('.card-text').textContent();
    check(text1.includes('发票'), '发票推荐');
    check((await card1.locator('.chip').count()) >= 1, '依据标签');
    check((await card1.locator('.card-meta').textContent()).includes('置信度'), '置信度');
    // 键盘：从提交按钮 Tab 到第一张卡片的第一个按钮
    await page.getByRole('button', { name: '获取推荐回答' }).focus();
    await page.keyboard.press('Tab');
    check((await page.evaluate(() => document.activeElement.textContent)) === '复制并标记已发送', 'Tab 顺序');
    await page.keyboard.press('Enter');
    await card1.locator('.status.ok').waitFor();
    check((await page.evaluate(() => navigator.clipboard.readText())) === text1, '剪贴板是推荐原文');
    check((await card1.getByRole('button').count()) === 0, '处理完收起按钮');
    steps.push('sent');

    // ---------------------------------------------------------------- 选商品 → 编辑后发送 → 候选知识
    await page.getByLabel('商品（选填）').selectOption('i_555');
    const turn2 = await ask('165 110斤穿什么码');
    check((await turn2.locator('.tag').textContent()).includes('法式碎花连衣裙'), '显示所选商品');
    const card2 = turn2.locator('article.card').first();
    await card2.getByRole('button', { name: '编辑后发送' }).click();
    const edited = '亲，165cm/110 斤建议选 M 码哦~';
    await card2.getByLabel('修改后的回答').fill(edited);
    await card2.getByRole('button', { name: '复制并提交修改' }).click();
    await card2.locator('.status.ok').filter({ hasText: '候选知识' }).waitFor();
    check((await card2.locator('.card-text').textContent()) === edited, '卡片显示实际发出的文字');
    check((await page.evaluate(() => navigator.clipboard.readText())) === edited, '剪贴板是修改后的文字');
    steps.push('edited');

    // ---------------------------------------------------------------- 不合适：理解错误 + 真实问题类型；Esc 取消
    await page.getByLabel('商品（选填）').selectOption('');
    const turn3 = await ask('什么时候发货');
    const card3 = turn3.locator('article.card').first();
    await card3.getByRole('button', { name: '不合适' }).click();
    await page.keyboard.press('Escape');
    check(await card3.getByRole('button', { name: '不合适' }).isVisible(), 'Esc 收起原因面板');
    await card3.getByRole('button', { name: '不合适' }).click();
    await card3.getByRole('button', { name: '提交' }).click();
    await card3.locator('.status.error').filter({ hasText: '请选择' }).waitFor();
    check(await card3.getByLabel('买家真正问的是').isHidden(), '没选理解错误时不问真实类型');
    await card3.getByLabel('理解错误').check();
    const intents = await card3.getByLabel('买家真正问的是').locator('option').evaluateAll((os) => os.map((o) => o.value));
    check(!intents.includes('shipping_policy') && intents.includes('urge_ship'), '真实类型不列推荐自己的类型');
    await card3.getByLabel('买家真正问的是').selectOption('urge_ship');
    await card3.getByRole('button', { name: '提交' }).click();
    await card3.locator('.status.ok').filter({ hasText: '标注集' }).waitFor();
    steps.push('rejected');

    // ---------------------------------------------------------------- 风险提示；追问后上一条过期
    const turn4 = await ask('能便宜点吗');
    await turn4.locator('.risks').filter({ hasText: '价格承诺' }).waitFor();
    await ask('有赠品吗');
    const expired = page.locator('article.turn').nth(1).locator('article.card').first();
    await expired.locator('.status').filter({ hasText: '已过期' }).waitFor();
    check((await expired.getByRole('button').count()) === 0, '过期卡片没有按钮');
    steps.push('expired');

    // ---------------------------------------------------------------- 新会话（换一位买家）：没有推荐 + 缺失知识说明
    // （同一会话里客服没回复的上一条会和这条合并理解，所以换个会话）
    const oldSession = await page.locator('#session-label').textContent();
    await page.getByRole('button', { name: '新会话（换一位买家）' }).click();
    check((await page.locator('#turns > *').count()) === 0, '新会话清空记录');
    check((await page.locator('#session-label').textContent()) !== oldSession, '新会话编号');
    const turn6 = await ask('这个面料是什么');
    await turn6.locator('.no-suggestion').waitFor();
    await turn6.locator('.missing').filter({ hasText: '会话里没有识别到当前商品' }).waitFor();
    steps.push('missing');

    // ---------------------------------------------------------------- XSS：买家问题原样当文字显示
    const evil = '<img src=x onerror="window.__xss=1">能开发票吗';
    const turn7 = await ask(evil);
    check((await turn7.locator('blockquote').textContent()) === evil, '原样显示');
    check((await page.locator('#turns img').count()) === 0, '没有生成标签');
    check((await page.evaluate(() => window.__xss)) === undefined, '没有执行');
    steps.push('xss');

    // ---------------------------------------------------------------- 影子模式、503（拦截接口响应）
    await page.route('**/standalone/ask', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          shop_id: 'x', conversation_id: 'standalone:x', mode: 'shadow', shadow: true, suggestions: [],
          alerts: [], no_suggestion_reason: null, missing_knowledge: [], missing_knowledge_details: [],
          auto_reply: null, transfer_to_human: false, degraded: [],
        }),
      }),
    );
    const shadowTurn = await ask('在吗');
    await shadowTurn.locator('.shadow').filter({ hasText: '影子模式' }).waitFor();
    check((await shadowTurn.locator('article.card').count()) === 0, '影子模式不显示卡片');
    await page.unroute('**/standalone/ask');
    await page.route('**/standalone/ask', (route) =>
      route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'storage_unavailable', message: '存储暂时不可用，请稍后重试' }),
      }),
    );
    const failed = await ask('在吗');
    await failed.getByRole('alert').filter({ hasText: '503' }).waitFor();
    check((await question.inputValue()) === '在吗', '失败时保留问题，方便重试');
    await page.unroute('**/standalone/ask');
    steps.push('shadow+503');

    // ---------------------------------------------------------------- 手机宽度没有横向滚动
    check(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      '手机宽度没有横向滚动',
    );
    steps.push('phone-width');

    // ---------------------------------------------------------------- 退出
    await page.getByRole('button', { name: '退出' }).click();
    await page.getByLabel('访问令牌').waitFor();
    check((await page.evaluate(() => sessionStorage.length)) === 0, '退出清掉 sessionStorage');
    check(await page.locator('#assist').isHidden(), '退出后隐藏工具');
    steps.push('sign-out');

    // ---------------------------------------------------------------- 商家：报表、缺失知识、问卷；刷新后保持登录
    await login(TOKENS.merchant);
    await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
    await page.locator('#adoption .headline').waitFor();
    check((await page.locator('#adoption .headline').textContent()).includes('%'), '采纳率');
    check((await page.locator('#adoption table tbody tr').count()) >= 1, '按问题类型');
    await page.locator('#gaps li').first().waitFor();
    await page.locator('#questionnaire summary').click();
    await page.getByLabel('第 11 题', { exact: false }).fill('保修一年，联系客服寄回维修');
    await page.getByRole('button', { name: '提交问卷' }).click();
    await page.locator('#questionnaire-status').filter({ hasText: '已保存' }).waitFor();
    await page.getByLabel('统计范围').selectOption('');
    await page.getByRole('button', { name: '刷新' }).click();
    await page.locator('#adoption .headline').waitFor();
    await page.reload();
    await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
    check(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      '报表在手机宽度下没有横向滚动',
    );
    steps.push('merchant');

    // ---------------------------------------------------------------- 令牌从不出现在网址里
    const secrets = Object.values(TOKENS);
    check(!urls.some((u) => secrets.some((t) => u.includes(t))), '令牌不进网址');
    check(problems.length === 0, `浏览器报错：${problems.join(' | ')}`);
    process.stdout.write(`${JSON.stringify({ ok: true, steps, requests: urls.length })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stdout.write(`${JSON.stringify({ ok: false, steps, error: String(err && err.stack ? err.stack : err) })}\n`);
  process.exit(1);
});
