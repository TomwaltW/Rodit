/*
 * 商家控制台「报表」页的平台客服指标一节（§12.4 接入前后对比）的真浏览器冒烟测试
 * （由 test_web_platform_metrics.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌 {manager, merchant}；
 * RODIT_SMOKE_GO_LIVE 服务端今天往前 28 天（接入日期未知时页面先按它算）；
 * RODIT_SMOKE_RECORD_DAY 商家登记用的日期。
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
const GO_LIVE = process.env.RODIT_SMOKE_GO_LIVE;
const RECORD_DAY = process.env.RODIT_SMOKE_RECORD_DAY;
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
    // 手机宽度：表格、图表都不能把页面撑出横向滚动
    const context = await browser.newContext({ viewport: { width: 375, height: 800 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      // 故意触发的 422 会有 "Failed to load resource"，不算问题；CSP 拦截、脚本报错算
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    page.on('request', (req) => urls.push(req.url()));

    const tab = (name) => page.getByRole('tab', { name, exact: true });
    const section = page.locator('.pm-section');
    const report = page.locator('#platform-metrics');
    const noHorizontalScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
    async function login(token) {
      await page.getByLabel('访问令牌').fill(token);
      await page.getByRole('button', { name: '登录', exact: true }).click();
    }
    async function signOut() {
      await page.getByRole('button', { name: '退出' }).click();
      await page.getByLabel('访问令牌').waitFor();
      check((await report.textContent()) === '', '退出后不留上一位的报表');
      check((await page.locator('#pm-status').textContent()) === '', '退出后清掉同步结果');
      check((await page.locator('#pm-go-live').inputValue()) === '', '退出后清掉接入日期');
    }

    // ---------------------------------------------------------------- 店长：还没有数据 → 同步 → 对比
    await page.goto(`${BASE}/console`);
    await login(TOKENS.manager);
    await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
    await tab('报表').click();
    await section.getByRole('heading', { name: '平台客服指标 · 接入前后对比' }).waitFor();
    // 演示单元没有上线记录、授权也没有开通时间：先按 28 天前算，提示填写
    await report.locator('.pm-guess').filter({ hasText: '接入日期未知' }).waitFor();
    check((await report.locator('.pm-guess').textContent()).includes(GO_LIVE), '先按 28 天前计算');
    check((await page.locator('#pm-go-live').inputValue()) === GO_LIVE, '接入日期框里是先用的日期');
    await report.locator('.headline').filter({ hasText: '数据不足' }).waitFor();
    check(await page.locator('#pm-record').isHidden(), '店长不能登记');
    check(await page.getByRole('button', { name: '同步平台数据' }).isVisible(), '店长可以同步');
    steps.push('empty');

    await page.getByRole('button', { name: '同步平台数据' }).click();
    await page.locator('#pm-status.ok').filter({ hasText: '写入 56 条' }).waitFor();
    await report.locator('.headline').filter({ hasText: '较接入前改善' }).waitFor();
    const row = report.locator('tbody tr').filter({ hasText: '旺旺 3 分钟人工响应率' });
    await row.waitFor();
    const change = row.locator('.pm-change.good');
    check((await change.textContent()).startsWith('↑'), '变化：向上的箭头');
    check((await change.textContent()).includes('个百分点'), '比例的变化按百分点');
    check((await row.textContent()).includes('平台数据 28 天'), '数据来源和天数');
    check((await row.textContent()).includes('越高越好'), '好坏方向');
    check((await report.locator('.pm-metric .hint').first().textContent()).startsWith('口径：'), '口径说明');
    // 按周的柱状图：接入前 4 周、接入后 4 周，接入日一条竖线，读屏有概要
    const chart = report.locator('.pm-chart svg');
    check((await chart.locator('.pm-bar.before').count()) === 4, '接入前 4 根柱子');
    check((await chart.locator('.pm-bar.after').count()) === 4, '接入后 4 根柱子');
    check((await chart.locator('.pm-golive').count()) === 1, '接入日的竖线');
    check((await chart.getAttribute('role')) === 'img', '图表 role=img');
    check((await chart.getAttribute('aria-label')).includes('接入前'), '图表有读屏概要');
    check((await chart.locator('.pm-week title').first().textContent()).includes('接入前第 4 周'), '每周有提示');
    check((await report.locator('.pm-legend').textContent()).includes('接入后'), '图例');
    // 柱子的颜色来自样式表（CSP 不许内联样式）：接入后和接入前颜色不同
    const fills = await chart.evaluate((svg) => [
      getComputedStyle(svg.querySelector('.pm-bar.before')).fill,
      getComputedStyle(svg.querySelector('.pm-bar.after')).fill,
    ]);
    check(fills[0] !== fills[1] && fills.every((f) => f && f !== 'none'), `柱子颜色：${fills}`);
    check(await noHorizontalScroll(), '报表在手机宽度下没有横向滚动');
    steps.push('sync');

    // 换成每段 14 天：图表只有 4 根柱子
    await page.getByLabel('对比天数').selectOption('14');
    await page.getByRole('button', { name: '按接入日期对比' }).click();
    await report.locator('.hint').filter({ hasText: '各 14 天' }).waitFor();
    check((await report.locator('.pm-chart .pm-bar').count()) === 4, '14 天：4 根柱子');
    // 填一个晚于今天的日期：服务端拒绝，页面说明原因
    await page.getByLabel('接入日期').fill('2099-01-01');
    await page.getByRole('button', { name: '按接入日期对比' }).click();
    await report.locator('.error-inline').filter({ hasText: '不能晚于今天' }).waitFor();
    // 填实际的接入日期：按填写的算
    await page.getByLabel('接入日期').fill(GO_LIVE);
    await page.getByRole('button', { name: '按接入日期对比' }).click();
    await report.locator('.hint').filter({ hasText: '按填写的接入日期' }).waitFor();
    check((await report.locator('.pm-guess').count()) === 0, '填了日期就不再提示未知');
    await signOut();
    steps.push('go-live');

    // ---------------------------------------------------------------- 商家：登记平台后台的数据
    await login(TOKENS.merchant);
    await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
    await tab('报表').click();
    await report.locator('.headline').filter({ hasText: '较接入前改善' }).waitFor();
    check(await page.locator('#pm-record').isVisible(), '商家可以登记');
    await page.locator('#pm-record summary').click();
    check((await page.locator('#pm-value-unit').textContent()).includes('%'), '比例按百分数填');
    await page.getByLabel('指标', { exact: true }).selectOption('taobao_3min_response_rate');
    await page.getByLabel('日期', { exact: true }).fill(RECORD_DAY);
    await page.getByRole('button', { name: '登记', exact: true }).click();
    await page.locator('#pm-record-status.error').filter({ hasText: '数值' }).waitFor();
    await page.getByLabel('数值').fill('93.5');
    await page.getByLabel('样本量（可不填）').fill('240');
    await page.getByRole('button', { name: '登记', exact: true }).click();
    await page.locator('#pm-record-status.ok').filter({ hasText: '已登记' }).waitFor();
    check((await page.getByLabel('数值').inputValue()) === '', '登记后清空数值');
    // 填成了 0–1 之外的百分数：服务端拒绝
    await page.getByLabel('数值').fill('150');
    await page.getByRole('button', { name: '登记', exact: true }).click();
    await page.locator('#pm-record-status.error').filter({ hasText: '422' }).waitFor();
    await page.getByLabel('数值').fill('93.5');
    await page.getByLabel('样本量（可不填）').fill('240');
    await page.getByRole('button', { name: '登记', exact: true }).click();
    await page.locator('#pm-record-status.ok').filter({ hasText: '没有改动' }).waitFor();
    check(await noHorizontalScroll(), '登记表单在手机宽度下没有横向滚动');
    await signOut();
    steps.push('record');

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
