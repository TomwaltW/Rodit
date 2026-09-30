/*
 * 话术助手页脚的生成式人工智能公示（方案 §12.3：公示所用模型的名称和备案号）的真浏览器冒烟测试
 * （由 test_disclosure.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌 {agent}；
 * RODIT_DISCLOSURE 期望显示的 {public_name, filing_no}（合成值）。
 * 退出码：0 通过；77 没有 Playwright / 浏览器起不来（Python 端跳过）；其他为失败。
 * 标准输出最后一行是 JSON 结果（steps 为走过的步骤）。
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
const EXPECTED = JSON.parse(process.env.RODIT_DISCLOSURE || '{}');
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
  try {
    const context = await browser.newContext({ viewport: { width: 375, height: 740 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      // CSP 拦截、脚本报错都会出现在这里
      if (msg.type() === 'error' || msg.type() === 'warning') problems.push(`console.${msg.type()}: ${msg.text()}`);
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));

    const response = await page.goto(`${BASE}/assist`);
    const csp = response.headers()['content-security-policy'] || '';
    check(csp.includes("default-src 'none'") && !csp.includes('unsafe'), 'CSP 没有放宽');
    const footer = page.locator('#disclosure');
    check(await footer.isHidden(), '登录前不显示公示（页面本身不含数据）');
    steps.push('page');

    await page.getByLabel('访问令牌').fill(TOKENS.agent);
    await page.getByRole('button', { name: '登录' }).click();
    await page.locator('#identity').filter({ hasText: '客服' }).waitFor();
    await footer.waitFor({ state: 'visible' });
    check((await page.locator('#llm-public-name').textContent()) === EXPECTED.public_name, '模型名称');
    check((await page.locator('#llm-filing-no').textContent()) === EXPECTED.filing_no, '备案号');
    const text = await footer.textContent();
    check(text.includes('生成式人工智能'), '说明推荐由生成式人工智能生成');
    check(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
      '手机宽度下没有横向滚动',
    );
    steps.push('shown');

    await page.getByRole('button', { name: '退出' }).click();
    await page.getByLabel('访问令牌').waitFor();
    check(await footer.isHidden(), '退出后不再显示');
    check((await page.locator('#llm-public-name').textContent()) === '', '退出后清掉');
    steps.push('signed-out');

    check(problems.length === 0, `浏览器报错：${problems.join(' | ')}`);
    process.stdout.write(`${JSON.stringify({ ok: true, steps })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stdout.write(`${JSON.stringify({ ok: false, steps, error: String(err && err.stack ? err.stack : err) })}\n`);
  process.exit(1);
});
