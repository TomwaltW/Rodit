/*
 * 售后审核工作台的真浏览器冒烟测试（由 test_web_aftersale_browser.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌
 * {agent, manager, reviewer, integration}；RODIT_SMOKE_XSS 写进 as_2007 买家说明的文字。
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
const XSS = process.env.RODIT_SMOKE_XSS || '';
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
  const posts = [];
  try {
    // 手机宽度：两栏变一栏，不能出现横向滚动
    const context = await browser.newContext({ viewport: { width: 375, height: 740 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      // 故意触发的 401 / 403 / 409 会有 "Failed to load resource"，不算问题；CSP 拦截、脚本报错算
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    page.on('request', (req) => {
      urls.push(req.url());
      if (req.method() === 'POST') posts.push(new URL(req.url()).pathname);
    });

    const noHorizontalScroll = () =>
      page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);

    async function login(token) {
      await page.getByLabel('访问令牌').fill(token);
      await page.getByRole('button', { name: '登录' }).click();
    }
    const item = (id) => page.locator(`#queue .queue-item[data-id="${id}"]`);
    async function open(id) {
      await item(id).click();
      await page.locator('#detail h3').filter({ hasText: `售后单 ${id}` }).waitFor();
      await page.locator('#detail[aria-busy="true"]').waitFor({ state: 'detached' }).catch(() => {});
    }
    const detailStatus = page.locator('#detail-status');

    // ---------------------------------------------------------------- 页面
    await page.goto(`${BASE}/aftersale`);
    check((await page.locator('#platform-name').textContent()) === '淘宝', '平台名');
    check((await page.locator('#platform-note').textContent()).includes('协商优先'), '淘宝拒绝要先协商的提示');
    const links = await page.locator('a.page-link').evaluateAll((as) => as.map((a) => a.getAttribute('href')));
    check(JSON.stringify(links) === JSON.stringify(['assist', 'console']), '页头链接到话术助手和控制台');
    steps.push('page');

    // ---------------------------------------------------------------- 登录：错误令牌、平台对接、复核人员
    await login('wrong-token-0000000000000000');
    await page.locator('#error').filter({ hasText: '401' }).waitFor();
    await login(TOKENS.integration);
    await page.locator('#error').filter({ hasText: '平台对接' }).waitFor();
    check(await page.locator('#app').isHidden(), '登录失败时不显示工作台');
    await login(TOKENS.reviewer);
    await page.locator('#error').filter({ hasText: '商家控制台' }).waitFor();
    check((await page.locator('#error a').getAttribute('href')) === 'console', '复核人员去控制台');
    check(await page.locator('#app').isHidden(), '复核人员看不到工作台');
    await page.locator('#identity').filter({ hasText: '复核人员' }).waitFor();
    await page.getByRole('button', { name: '退出' }).click();
    check((await page.evaluate(() => sessionStorage.length)) === 0, '退出清掉令牌');
    steps.push('login-guards');

    // ---------------------------------------------------------------- 客服登录：列表、概况
    await login(TOKENS.agent);
    await page.locator('#identity').filter({ hasText: '客服' }).waitFor();
    await item('as_2001').waitFor();
    check((await page.locator('#queue .queue-item').count()) === 8, '平台上待审核的 8 张');
    check((await item('as_2007').textContent()).includes('未跑规则'), '还没跑规则');
    check((await page.locator('#queue .queue-item').first().getAttribute('data-id')) === 'as_2007', '等得最久的在前');
    await page.locator('#stats .tile').first().waitFor();
    check((await page.locator('#stats').textContent()).includes('未跑规则8'), '概况：未跑规则 8 张');
    // 买家说明原样当文字显示（XSS）
    check((await item('as_2007').textContent()).length > 0, '列表行');
    check((await page.locator('#queue img').count()) === 0, '列表里没有生成标签');
    steps.push('queue');

    // ---------------------------------------------------------------- 拉取并跑规则
    await page.getByRole('button', { name: '拉取并跑规则' }).click();
    await page.locator('#run-status').filter({ hasText: '成功 8 张' }).waitFor();
    await item('as_2001').filter({ hasText: '待客服确认' }).waitFor();
    await item('as_2005').filter({ hasText: '须店长确认' }).waitFor();
    await item('as_3004').filter({ hasText: '风险' }).waitFor();
    check(posts.filter((p) => p.endsWith('/aftersales/review-pending')).length === 1, '批量跑规则调了一次');
    steps.push('run-rules');

    // ---------------------------------------------------------------- 签收核查中（R4）：不能执行，原因写清楚
    await open('as_2003');
    const current = page.locator('#detail .state-strip li[aria-current="step"]');
    check((await current.textContent()).startsWith('签收核查中'), '状态条高亮签收核查中');
    check((await page.locator('#detail .state-strip li.visited').count()) >= 2, '经过的状态打勾');
    check((await page.locator('#detail .rules').textContent()).includes('物流显示签收，买家称没收到'), 'R4 说明');
    for (const name of ['同意退款', '同意退货', '拒绝售后']) {
      check(await page.getByRole('button', { name, exact: true }).isDisabled(), `${name} 不能点`);
    }
    check((await page.locator('#detail .why').first().textContent()).includes('签收核查中'), '不能执行的原因');
    check((await page.locator('#detail .buyer-note').textContent()) === '显示签收了但我根本没收到', '买家说明');
    steps.push('verify');

    // ---------------------------------------------------------------- R6：客服不能确认
    await open('as_2005');
    const managerNote = page.locator('#manager-note');
    check((await managerNote.textContent()).includes('必须店长确认（R6）'), 'R6 提示');
    check((await managerNote.textContent()).includes('不能确认'), '客服不能确认');
    const r6 = page.getByRole('button', { name: '同意退货（建议）' });
    check(await r6.isDisabled(), '客服的确认按钮禁用');
    const why = await page.locator(`#${await r6.getAttribute('aria-describedby')}`).textContent();
    check(why.includes('店长'), '按钮说明为什么不能点');
    steps.push('manager-guard');

    // ---------------------------------------------------------------- XSS：买家说明原样当文字
    await open('as_2007');
    check((await page.locator('#detail .buyer-note').textContent()) === XSS, '原样显示');
    check((await page.locator('#detail img, #queue img').count()) === 0, '没有生成标签');
    check((await page.evaluate(() => window.__xss)) === undefined, '没有执行');
    steps.push('xss');

    // ---------------------------------------------------------------- 协商 → 协商有结果
    await page.getByRole('button', { name: '协商', exact: true }).click();
    await page.getByLabel('协商备注（选填）').fill('先和买家沟通，电话 13812345678');
    await page.getByRole('button', { name: '提交协商', exact: true }).click();
    await detailStatus.filter({ hasText: '已进入协商' }).waitFor();
    await page.locator('#detail .detail-head').filter({ hasText: '协商中' }).waitFor();
    await page.getByRole('button', { name: '协商有结果', exact: true }).click();
    await page.getByLabel('协商结果（选填）').fill('买家坚持退货');
    await page.getByRole('button', { name: '提交协商有结果' }).click();
    await detailStatus.filter({ hasText: '协商已记录' }).waitFor();
    await page.getByRole('button', { name: '拒绝售后', exact: true }).waitFor();
    check(!(await page.getByRole('button', { name: '拒绝售后', exact: true }).isDisabled()), '淘宝协商过后才能拒绝');
    check((await page.locator('#detail .history').textContent()).includes('协商中'), '处理记录');
    check(!(await page.locator('#detail .history').textContent()).includes('13812345678'), '备注打码');
    steps.push('negotiate');

    // ---------------------------------------------------------------- 状态已变化：提示并自动刷新
    await open('as_2002');
    await page.route('**/aftersales/as_2002/confirm', (route) =>
      route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'action_not_allowed', message: '售后单状态已变化，请刷新后重新确认' }),
      }),
    );
    const before = posts.length;
    await page.getByRole('button', { name: '同意退货（建议）' }).click();
    await page.locator('#confirm-dialog').waitFor({ state: 'visible' });
    await page.getByRole('button', { name: '确认执行' }).click();
    await detailStatus.filter({ hasText: '售后单状态已变化，请刷新' }).waitFor();
    await detailStatus.filter({ hasText: '已刷新' }).waitFor();
    check(await page.locator('#confirm-dialog').isHidden(), '对话框关闭');
    check(posts.slice(before).some((p) => p.endsWith('/aftersales/as_2002/review')), '自动刷新（重读平台）');
    await page.unroute('**/aftersales/as_2002/confirm');
    steps.push('stale');

    // ---------------------------------------------------------------- 确认 as_2001（R2）：已同意
    await open('as_2001');
    const agree = page.getByRole('button', { name: '同意退款（建议）' });
    check(!(await agree.isDisabled()), '客服可以确认 R2');
    await agree.click();
    const dialog = page.locator('#confirm-dialog');
    await dialog.waitFor({ state: 'visible' });
    check((await page.locator('#confirm-rule').textContent()).includes('必须由人确认'), '对话框说明人工确认');
    check((await page.locator('#confirm-facts').textContent()).includes('状态版本'), '对话框显示状态版本');
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'hidden' });
    check(await agree.evaluate((b) => b === document.activeElement), 'Esc 关闭后焦点回到按钮');
    await agree.click();
    await dialog.waitFor({ state: 'visible' });
    check(await noHorizontalScroll(), '对话框在手机宽度下没有横向滚动');
    await page.getByRole('button', { name: '确认执行' }).click();
    await detailStatus.filter({ hasText: '已执行同意退款' }).waitFor();
    await page.locator('#detail .detail-head').filter({ hasText: '已同意' }).waitFor();
    await page.locator('#detail .result-line.ok').waitFor();
    await item('as_2001').waitFor({ state: 'detached' });
    await page.locator('#stats .tile').filter({ hasText: '平均审核时长' }).filter({ hasNotText: '—' }).waitFor();
    steps.push('confirm');

    // ---------------------------------------------------------------- 筛选：已同意
    await page.getByLabel('审核状态').selectOption('agreed');
    await item('as_2001').waitFor();
    check((await page.locator('#queue .queue-item').count()) === 1, '已同意的只有 as_2001');
    await page.getByLabel('审核状态').selectOption('');
    await item('as_2007').waitFor();
    steps.push('filter');

    check(await noHorizontalScroll(), '手机宽度没有横向滚动');
    steps.push('phone-width');

    // ---------------------------------------------------------------- 退出；店长可以确认 R6
    await page.getByRole('button', { name: '退出' }).click();
    await page.getByLabel('访问令牌').waitFor();
    check((await page.evaluate(() => sessionStorage.length)) === 0, '退出清掉 sessionStorage');
    check(await page.locator('#app').isHidden(), '退出后隐藏工作台');
    await login(TOKENS.manager);
    await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
    await item('as_2005').waitFor();
    await open('as_2005');
    check((await page.locator('#manager-note').textContent()).includes('你是店长'), '店长提示');
    check(!(await page.getByRole('button', { name: '同意退货（建议）' }).isDisabled()), '店长可以确认 R6');
    await page.reload();
    await page.locator('#identity').filter({ hasText: '店长' }).waitFor(); // 刷新后保持登录
    steps.push('manager');

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
