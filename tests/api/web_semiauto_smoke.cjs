/*
 * 半自动与全自动生效时段的真浏览器冒烟测试（由 test_web_semiauto.py 启动；Chromium + Playwright）。
 * 方案 §5.6 半自动、§13.4 "该类型原样采纳率 ≥ 90%"、§15 Phase 3 全自动先在非工作时间试行。
 *
 * - 话术助手页：开了半自动的类型，推荐卡片上有"半自动：可直接发送"标记，客服照常点发送；
 *   没开的类型没有标记；
 * - 控制台「上线」页：全自动生效时段（只读）、各类型的半自动门槛；店长只读，商家只能关闭，
 *   复核人员门槛达到直接开、没达到写原因强制开。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌
 * {agent, manager, merchant, reviewer}。退出码：0 通过；77 没有 Playwright / 浏览器起不来（Python
 * 端跳过）；其他为失败。标准输出最后一行是 JSON 结果（steps 为走过的步骤）。数据全部是合成的。
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
    // 手机宽度：新加的表格、提示都不能把页面撑出横向滚动
    const context = await browser.newContext({ viewport: { width: 375, height: 800 }, locale: 'zh-CN' });
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: BASE });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      // 故意触发的 409 会有 "Failed to load resource"，不算问题；CSP 拦截、脚本报错算
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    page.on('request', (req) => urls.push(req.url()));

    const noHorizontalScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
    async function login(token) {
      await page.getByLabel('访问令牌').fill(token);
      await page.getByRole('button', { name: '登录', exact: true }).click();
    }
    async function signOut() {
      await page.getByRole('button', { name: '退出' }).click();
      await page.getByLabel('访问令牌').waitFor();
    }

    // ---------------------------------------------------------------- 话术助手：半自动标记
    const question = page.getByLabel('买家的问题');
    const newestTurn = () => page.locator('article.turn').first();
    async function ask(text) {
      await question.fill(text);
      await page.getByRole('button', { name: '获取推荐回答' }).click();
      await newestTurn().locator('.turn-body').filter({ hasNotText: '正在生成推荐' }).waitFor();
      return newestTurn();
    }

    await page.goto(`${BASE}/assist`);
    await login(TOKENS.agent);
    await page.locator('#identity').filter({ hasText: '客服' }).waitFor();
    const turn1 = await ask('什么时候发货');
    const card1 = turn1.locator('article.card').first();
    await card1.waitFor();
    await card1.locator('.badge.prefill').filter({ hasText: '半自动：可直接发送' }).waitFor();
    check(await card1.evaluate((n) => n.classList.contains('prefill')), '卡片整体标出半自动');
    check(
      (await card1.locator('.card-meta .badge').first().textContent()) === '半自动：可直接发送',
      '半自动标记放在最前面',
    );
    check((await card1.locator('.prefill-hint').textContent()).includes('仍由你来发'), '提示仍由客服发送');
    check((await card1.locator('.prefill-hint').getAttribute('style')) === null, '没有行内样式（CSP）');
    // 发送和平时一样：复制并标记已发送（不是自动发送）
    const text1 = await card1.locator('.card-text').textContent();
    await card1.getByRole('button', { name: '复制并标记已发送' }).click();
    await card1.locator('.status.ok').waitFor();
    check((await page.evaluate(() => navigator.clipboard.readText())) === text1, '剪贴板是推荐原文');
    check(await card1.evaluate((n) => n.classList.contains('done')), '发送后卡片收起');
    // 换一位买家，问没开半自动的类型：没有标记
    await page.getByRole('button', { name: '新会话（换一位买家）' }).click();
    const turn2 = await ask('能开发票吗');
    await turn2.locator('article.card').first().waitFor();
    check((await turn2.locator('.badge.prefill').count()) === 0, '没开半自动的类型没有标记');
    check((await turn2.locator('article.card.prefill').count()) === 0, '没开半自动的卡片不加框');
    check(await noHorizontalScroll(), '话术助手页在手机宽度下没有横向滚动');
    await signOut();
    steps.push('assist-badge');

    // ---------------------------------------------------------------- 控制台：店长只读
    const tab = (name) => page.getByRole('tab', { name, exact: true });
    const semi = page.locator('#semi-auto');
    const semiStatus = page.locator('#semi-auto-status');
    const enabledLine = page.locator('#semi-auto-enabled');

    await page.goto(`${BASE}/console`);
    await login(TOKENS.manager);
    await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
    await tab('上线').click();
    const schedule = page.locator('#auto-schedule');
    await schedule.waitFor();
    const scheduleText = await schedule.textContent();
    check(scheduleText.includes('09:00-23:00'), '显示客服工作时间');
    check(scheduleText.includes('北京时间'), '写明北京时间');
    check(scheduleText.includes('只在客服工作时间以外'), '显示生效时段');
    check(scheduleText.includes('客服工作时间内'), '显示现在在工作时间内');
    check(scheduleText.includes('暂停：现在是客服工作时间'), '全自动现在暂停，并说明原因');
    check(scheduleText.includes('2026-09-28 23:00'), '下一次进出工作时间');
    await semi.waitFor();
    check((await enabledLine.textContent()).includes('发货与快递'), '显示已开的类型');
    check((await semi.getByRole('button').count()) === 0, '店长没有半自动的开关');
    check((await semi.textContent()).includes('价格 / 优惠（高风险类型，不能开）'), '高风险类型写明不能开');
    const sizeRow = semi.locator('tr[data-intent="size_advice"]');
    check((await sizeRow.textContent()).includes('没达到'), '门槛没达到的类型');
    const invoiceRow = semi.locator('tr[data-intent="invoice_gift"]');
    check((await invoiceRow.textContent()).includes('达到'), '门槛达到的类型');
    check((await invoiceRow.textContent()).includes('第 2 周 95%'), '按周显示原样采纳率');
    check(await noHorizontalScroll(), '上线页在手机宽度下没有横向滚动（店长）');
    await signOut();
    steps.push('manager');

    // ---------------------------------------------------------------- 商家：在设置页填客服工作时间
    await login(TOKENS.merchant);
    await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
    await tab('设置').click();
    const hoursRow = page.locator('.config-row[data-key="service_hours"]');
    await hoursRow.waitFor();
    check((await hoursRow.textContent()).includes('09:00-23:00'), '设置页显示现在的工作时间');
    const hoursBox = hoursRow.getByLabel('修改客服工作时间');
    await hoursBox.fill('9点到11点');
    await hoursRow.getByRole('button', { name: '保存' }).click();
    await hoursRow.locator('.status.error').filter({ hasText: '上午还是晚上' }).waitFor();
    await hoursBox.fill('9:00-12:00\n下午1点半到晚10点');
    await hoursRow.getByRole('button', { name: '保存' }).click();
    const savedRow = page.locator('.config-row[data-key="service_hours"]');
    await savedRow.locator('.status.ok').filter({ hasText: '09:00-12:00、13:30-22:00' }).waitFor();
    steps.push('service-hours');

    // ---------------------------------------------------------------- 商家：半自动只能关闭
    await tab('上线').click();
    await schedule.filter({ hasText: '09:00-12:00、13:30-22:00' }).waitFor();
    check((await schedule.textContent()).includes('2026-09-28 12:00'), '下一次进出工作时间跟着变');
    await semi.waitFor();
    check((await semi.getByRole('button', { name: /^开启/ }).count()) === 0, '商家没有开启按钮');
    check((await semi.textContent()).includes('开启由复核人员确认门槛后操作'), '商家看得到谁能开启');
    await semi.getByRole('button', { name: '关闭「发货与快递」的半自动' }).click();
    await semiStatus.filter({ hasText: '已关闭「发货与快递」的半自动' }).waitFor();
    check(await semiStatus.evaluate((n) => n.classList.contains('ok')), '关闭成功的提示');
    await enabledLine.filter({ hasText: '已开半自动：没有' }).waitFor();
    await signOut();
    steps.push('merchant');

    // ---------------------------------------------------------------- 复核人员：达到门槛直接开；没达到写原因强制开
    await login(TOKENS.reviewer);
    await page.locator('#identity').filter({ hasText: '复核人员' }).waitFor();
    await tab('上线').click();
    await semi.waitFor();
    await semi.getByRole('button', { name: '开启「发票 / 赠品 / 保障」的半自动' }).click();
    await semiStatus.filter({ hasText: '已开启「发票 / 赠品 / 保障」的半自动' }).waitFor();
    check(!(await semiStatus.textContent()).includes('强制'), '门槛达到不是强制开启');
    await enabledLine.filter({ hasText: '发票 / 赠品 / 保障' }).waitFor();
    check(await semi.getByRole('button', { name: '关闭「发票 / 赠品 / 保障」的半自动' }).isVisible(), '开了以后可以关');

    await semi.getByRole('button', { name: '开启「尺码 / 选型」的半自动' }).click();
    const failure = semi.locator('.gate-failure');
    await failure.filter({ hasText: '门槛没过' }).waitFor();
    check((await failure.locator('tbody tr').count()) >= 3, '显示 409 带回的门槛结果');
    await failure.getByRole('button', { name: '取消' }).click();
    check((await semi.locator('.gate-failure').count()) === 0, '取消收起强制面板');
    await semi.getByRole('button', { name: '开启「尺码 / 选型」的半自动' }).click();
    await failure.filter({ hasText: '门槛没过' }).waitFor();
    await failure.getByRole('button', { name: '强制开启' }).click();
    await semiStatus.filter({ hasText: '写明原因' }).waitFor();
    check(await semiStatus.evaluate((n) => n.classList.contains('error')), '没写原因的提示');
    await failure.getByLabel('半自动的原因', { exact: false }).fill('种子商家同意先试');
    await failure.getByRole('button', { name: '强制开启' }).click();
    await semiStatus.filter({ hasText: '强制开启' }).waitFor();
    await enabledLine.filter({ hasText: '尺码 / 选型' }).waitFor();
    check((await semi.getByRole('button', { name: /^开启「价格/ }).count()) === 0, '高风险类型没有开启按钮');
    check(await noHorizontalScroll(), '上线页在手机宽度下没有横向滚动（复核人员）');
    await signOut();
    steps.push('reviewer');

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
