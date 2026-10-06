/*
 * 客户演示站接"真实大模型"时的真浏览器冒烟测试（由 test_browser.py 启动；Chromium + Playwright）。
 * 服务端的大模型是本机的 OpenAI 兼容替身（llm_stub.py），走真实的 HTTP，不联网。
 *
 * 核对：页眉下的徽标和页脚写明"真实大模型：<模型>（<主机>）· 判断层 · 时间预算"；第一条消息的路由
 * 三票都失败（Python 端让替身限流）时，过程面板写明退回了关键词规则；之后的消息过程面板显示真实的
 * 票数、生成由真实大模型写、真实调用的次数；§11.5 重写那次生成失败（替身对重写回 500）时，页面不说
 * "重写后通过"、写明生成失败退回 FAQ / 模板；手机宽度没有横向滚动；没有控制台报错、CSP 拦截、外部请求。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_LLM_MODEL 模型名；RODIT_EXPECT_HOST 接口主机；
 * RODIT_SHOWCASE_SHOTS（可选）截图目录。退出码：0 通过；77 没有 Playwright / 浏览器（跳过）；其他失败。
 * 标准输出最后一行是 JSON 结果（steps 为走过的步骤）。数据全部是合成的演示数据。
 */
'use strict';

const fs = require('fs');
const path = require('path');

const SKIP = 77;
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch (err) {
  process.stderr.write(`没有 playwright 模块：${err.message}\n`);
  process.exit(SKIP);
}

const BASE = process.env.RODIT_BASE_URL;
const MODEL = process.env.RODIT_LLM_MODEL;
const HOST = process.env.RODIT_EXPECT_HOST;
const SHOTS = process.env.RODIT_SHOWCASE_SHOTS || '';
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

async function shot(page, name) {
  if (!SHOTS) return;
  fs.mkdirSync(SHOTS, { recursive: true });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`), fullPage: true });
}

async function newPage(browser, viewport, problems) {
  const context = await browser.newContext({ viewport, locale: 'zh-CN' });
  await context.addInitScript(() => {
    window.__cspViolations = [];
    document.addEventListener('securitypolicyviolation', (e) => {
      window.__cspViolations.push(`${e.violatedDirective} ${e.blockedURI}`);
    });
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warning') problems.push(`console.${msg.type()}: ${msg.text()}`);
  });
  page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
  page.on('request', (req) => {
    if (!req.url().startsWith(BASE) && !req.url().startsWith('data:')) problems.push(`外部请求：${req.url()}`);
  });
  return page;
}

async function noHorizontalScroll(page, label) {
  const { scroll, inner } = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, inner: window.innerWidth }));
  check(scroll <= inner, `${label}：页面横向滚动（${scroll} > ${inner}）`);
}

async function cspClean(page, label) {
  const violations = await page.evaluate(() => window.__cspViolations || []);
  check(violations.length === 0, `${label}：CSP 拦截 ${violations.join('; ')}`);
}

async function main() {
  const browser = await launch();
  const problems = [];
  try {
    const page = await newPage(browser, { width: 1440, height: 900 }, problems);

    // ---------------------------------------------------------------- 徽标：这次演示用的是什么
    await page.goto(`${BASE}/`);
    await page.locator('#hero-title').waitFor();
    await page.locator('#chain-strip').waitFor();
    const badge = await page.locator('#chain-badge').textContent();
    const expected = `真实大模型：${MODEL}（${HOST}）· 判断层：大模型投票 · 时间预算：严格`;
    check(badge === expected, `页眉徽标：${badge}`);
    check(await page.locator('#chain-relaxed').isHidden(), '严格时间预算没有"放宽"说明');
    check((await page.locator('#footer-badge').textContent()) === expected, '页脚徽标');
    check((await page.locator('#footer-text').textContent()).includes(`回答由真实大模型 ${MODEL} 生成`), '页脚说明');
    await shot(page, 'llm-landing');
    steps.push('badge');

    // ---------------------------------------------------------------- 第一条消息：三票都被限流 → 关键词规则兜底
    await page.goto(`${BASE}/#/workbench`);
    await page.locator('#mode-label').filter({ hasText: '千牛' }).waitFor();
    await page.locator('.conv-btn[data-scenario="ship_quick"]').click();
    await page.locator('.sg-card').first().waitFor();
    await page.locator('#trace-toggle').click();
    await page.locator('#trace').waitFor();
    const fallback = await page.locator('#trace .fallback-note').first().textContent();
    check(fallback.includes('大模型 3 次调用全部失败') && fallback.includes('关键词规则路由'), `降级说明：${fallback}`);
    const failedBox = await page.locator('#trace .vote-box h4').first().textContent();
    check(failedBox.includes('0/3 次有回答'), `失败的投票：${failedBox}`);
    check((await page.locator('#trace .vote-box').first().textContent()).includes('限流或额度用完（HTTP 429） ×3'), '失败原因');
    check((await page.locator('#trace').textContent()).includes('关键词规则（大模型判断层不可用时的兜底）'), '判断层写明是兜底');
    await shot(page, 'llm-fallback');
    steps.push('fallback');

    // ---------------------------------------------------------------- 之后：真实的票数、真实生成
    await page.locator('.conv-btn[data-scenario="size"]').click();
    await page.locator('#trace .vote-box h4').filter({ hasText: '3/3 次有回答' }).first().waitFor();
    const trace = await page.locator('#trace').textContent();
    check(trace.includes(`大模型投票（${MODEL}）`), '路由由大模型投票');
    check(trace.includes('尺码 / 选型 ×3'), '三票都选尺码');
    check(trace.includes('无（只有一个问题） ×3'), '追问的票');
    check(trace.includes(`大模型草稿由真实大模型 ${MODEL} 生成`), '生成由真实大模型写');
    check(trace.includes(`真实大模型调用（${MODEL}）：路由 3 次`), '真实调用次数');
    check(trace.includes('时间预算按生产（§12.1）'), '时间预算说明');
    check(!trace.includes('离线替身不联网'), '没有离线替身的说明');
    check((await page.locator('.sg-card textarea').first().inputValue()).includes('M'), '尺码推荐');
    await cspClean(page, '桌面');
    await shot(page, 'llm-votes');
    steps.push('votes');

    // ---------------------------------------------------------------- §11.5 重写那次生成失败（Python 端让替身对重写回 500）
    await page.locator('.conv-btn[data-scenario="qa_block"]').click();
    await page.locator('#trace .generation-note').first().waitFor();
    const simulated = await page.locator('#ai-simulated').textContent();
    check(simulated.includes('没有大模型重写稿') && !simulated.includes('重写后通过'), `重写失败的说明：${simulated}`);
    const genNote = await page.locator('#trace .generation-note').first().textContent();
    check(genNote.includes('大模型生成有') && genNote.includes('服务端错误（HTTP 500）') && genNote.includes('§12.2'), `生成失败的降级说明：${genNote}`);
    check((await page.locator('#trace').textContent()).includes('生成 2 次（失败 2）'), '生成失败的次数');
    await shot(page, 'llm-rewrite-failed');
    steps.push('rewrite_failed');

    // ---------------------------------------------------------------- 手机宽度
    const phone = await newPage(browser, { width: 390, height: 844 }, problems);
    for (const route of ['overview', 'workbench']) {
      await phone.goto(`${BASE}/#/${route}`);
      await phone.locator(`#page-${route}`).waitFor();
      await phone.locator('#chain-strip').waitFor();
      if (route === 'workbench') {
        await phone.locator('.conv-btn[data-scenario="multi"]').click();
        await phone.locator('.sg-card').first().waitFor();
        await phone.locator('#trace-toggle').click();
        await phone.locator('#trace .vote-box').first().waitFor();
        await shot(phone, 'llm-phone-workbench');
      }
      await noHorizontalScroll(phone, `手机 ${route}`);
    }
    await cspClean(phone, '手机');
    steps.push('phone');

    check(problems.length === 0, problems.join('\n'));
    console.log(JSON.stringify({ ok: true, steps }));
  } catch (err) {
    console.log(JSON.stringify({ ok: false, steps, error: `${err.message}\n${problems.join('\n')}` }));
    process.exitCode = 1;
  } finally {
    await browser.close();
  }
}

main();
