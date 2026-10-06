/*
 * 客户演示站的真浏览器冒烟测试（由 test_browser.py 启动；Chromium + Playwright）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SHOWCASE_SHOTS（可选）截图保存目录。
 * 退出码：0 通过；77 没有 Playwright / 浏览器起不来（Python 端跳过）；其他为失败。
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
  // CSP 拦截的事件记下来（initScript 由浏览器注入，不受页面 CSP 限制）
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

    // ---------------------------------------------------------------- 产品概览
    await page.goto(`${BASE}/`);
    await page.locator('#hero-title').waitFor();
    check((await page.locator('.brand-name').textContent()) === 'Rodit', '品牌');
    check((await page.locator('.brand-tagline').textContent()).includes('淘宝 · 抖店 · 拼多多'), '标语');
    await page.locator('#targets .kpi').nth(5).waitFor();
    check((await page.locator('#targets .kpi').count()) === 6, '§1.3 六个目标');
    check((await page.locator('#targets .kpi[data-key="adoption"] .kpi-value').textContent()).endsWith('%'), '采纳率是百分比');
    check((await page.locator('.step-card').count()) === 5, '懂找写判兜底五步');
    check((await page.locator('.finding').count()) === 3, '三个发现');
    check((await page.locator('.main-nav a[aria-current="page"]').textContent()) === '产品概览', '导航高亮');
    check((await page.locator('#footer-text').textContent()).includes('合成数据'), '页脚');
    await shot(page, 'landing');
    steps.push('landing');

    // ---------------------------------------------------------------- 工作台：尺码咨询 → 推荐 + 过程
    await page.getByRole('link', { name: '进入智能辅助工作台' }).click();
    await page.locator('#page-workbench').waitFor();
    check((await page.locator('#mode-label').textContent()).includes('千牛'), '模式 B 千牛');
    await page.locator('.conv-btn[data-scenario="size"]').click();
    const card = page.locator('.sg-card').first();
    await card.waitFor();
    check((await card.locator('textarea').inputValue()).includes('M'), '尺码推荐');
    check((await card.locator('.chip').allTextContents()).includes('商品属性 · 尺码建议'), '依据标签');
    check((await card.locator('.badge').allTextContents()).some((t) => t.includes('置信度')), '置信度');
    check((await page.locator('.msg-buyer .bubble').first().textContent()) === '165 110斤穿什么码', '买家气泡');
    await page.locator('#trace-toggle').click();
    await page.locator('#trace').waitFor();
    check((await page.locator('#trace .trace-step').count()) === 6, '懂找写判兜底 + 耗时');
    check((await page.locator('#trace').textContent()).includes('§6.6 规则 6'), '规则说明');
    check((await page.locator('#trace .ev-label').allTextContents()).includes('商品属性 · 尺码建议'), '依据列表');
    check((await page.locator('#trace .prob-row').count()) >= 1, '概率条');
    steps.push('suggestion-trace');

    // 发送 → 客服气泡
    await card.locator('button[data-action="sent"]').click();
    await page.locator('.msg-agent .bubble').waitFor();
    check((await page.locator('#toast').textContent()).includes('原样发送'), '发送提示');
    check((await page.locator('.sg-card .sg-done').textContent()).includes('已原样发送'), '卡片状态');
    steps.push('send');

    // 质检拦截：第一稿被拦下、重写后通过
    await page.locator('.conv-btn[data-scenario="qa_block"]').click();
    await page.locator('#ai-simulated').waitFor();
    check((await page.locator('#ai-simulated').textContent()).includes('演示：模拟大模型先写出没有依据的承诺'), '模拟说明');
    await page.locator('#trace .check-fail').waitFor();
    check((await page.locator('#trace .check-fail .check-text').textContent()) === '亲，明天一定能到哦！', '被拦下的第一稿');
    check((await page.locator('#trace .check-pass').count()) >= 1, '重写后通过');
    await shot(page, 'workbench-trace');
    steps.push('qa-block');

    // 编辑后发送、不合适
    await page.locator('.conv-btn[data-scenario="invoice"]').click();
    const invoice = page.locator('.sg-card').first();
    await invoice.waitFor();
    await invoice.locator('textarea').fill('亲，可以开电子普通发票，确认收货后 3 个工作日内开出，有问题随时找我~');
    await invoice.locator('button[data-action="edited"]').click();
    await page.locator('#toast').filter({ hasText: '候选知识' }).waitFor();
    check((await page.locator('.msg-agent .bubble').last().textContent()).includes('有问题随时找我'), '编辑后的文字');
    await page.locator('.conv-btn[data-scenario="price"]').click();
    const price = page.locator('.sg-card').first();
    await price.waitFor();
    check((await price.locator('.risk').allTextContents()).includes('涉及价格承诺，请确认'), '风险提示');
    await price.locator('button[data-action="rejected"]').click();
    await price.locator('button[data-reason="bad_wording"]').click();
    await page.locator('#toast').filter({ hasText: '表达不好' }).waitFor();
    steps.push('feedback');

    // 投诉：不推荐，标红提醒
    await page.locator('.conv-btn[data-scenario="complaint"]').click();
    await page.locator('#ai-alerts .alert').first().waitFor();
    check((await page.locator('.no-suggestion').textContent()).includes('不推荐'), '投诉不推荐');
    // 买家打字（Ctrl+Enter）：回到尺码会话接着问（客服已经回复过，是新的一轮）
    await page.locator('.conv-btn[data-scenario="size"]').click();
    await page.locator('#composer-text').fill('那 170 120斤呢');
    await page.locator('#composer-text').press('Control+Enter');
    await page.locator('.msg-buyer .bubble').filter({ hasText: '那 170 120斤呢' }).waitFor();
    const next = page.locator('.sg-card[data-status="shown"]').first();
    await next.waitFor();
    check((await next.locator('textarea').inputValue()).includes('L'), '第二轮推荐 L 码');
    steps.push('typing');

    // ---------------------------------------------------------------- 售后审核：确认 → 重读 + 幂等
    await page.getByRole('navigation', { name: '主导航' }).getByRole('link', { name: '售后审核助手' }).click();
    const r2 = page.locator('.as-card[data-id="as_2001"]');
    await r2.waitFor();
    check((await r2.textContent()).includes('R2'), 'R2 规则');
    check((await page.locator('.as-card[data-id="as_2005"]').textContent()).includes('R6 须店长确认'), 'R6 徽标');
    check((await page.locator('#state-strip li').count()) === 10, '状态机');
    await r2.locator('button[data-action="agree_refund"]').click();
    const dialog = page.locator('#confirm-dialog');
    await dialog.waitFor();
    check((await dialog.textContent()).includes('售后写操作必须人确认'), '确认框说明');
    check((await dialog.textContent()).includes('as_2001:agree_refund:'), '幂等键');
    await page.locator('#dialog-ok').click();
    await r2.locator('.as-result').waitFor();
    const result = await r2.locator('.as-result').textContent();
    check(result.includes('已执行') && result.includes('待商家处理') && result.includes('退款中'), `重读状态：${result}`);
    check((await page.locator('.state-pill.is-current').textContent()) === '已同意', '状态条高亮');
    await r2.locator('button[data-action="repeat"]').click();
    await r2.locator('.as-result').filter({ hasText: '幂等' }).waitFor();
    // 取消不执行
    const r3 = page.locator('.as-card[data-id="as_2002"]');
    await r3.locator('button[data-action="agree_return"]').click();
    await dialog.waitFor();
    await page.locator('#dialog-cancel').click();
    check((await r3.locator('.as-result').count()) === 0, '取消就不执行');
    await shot(page, 'aftersale');
    steps.push('aftersale');

    // ---------------------------------------------------------------- 数据看板
    await page.getByRole('navigation', { name: '主导航' }).getByRole('link', { name: '数据看板' }).click();
    await page.locator('#chart-weekly svg').waitFor();
    check((await page.locator('#kpis .kpi').count()) === 6, '指标卡');
    check((await page.locator('#chart-weekly .series-dot').count()) === 4, '4 周采纳率（第 1 周影子模式）');
    check((await page.locator('#chart-intents .hbar').count()) >= 8, '问题分布');
    check((await page.locator('#chart-metrics .metric').count()) >= 1, '平台指标');
    check((await page.locator('#chart-metrics .delta-good').count()) >= 1, '接入后改善');
    check((await page.locator('#page-dashboard').textContent()).includes('演示数据（合成）'), '合成数据标注');
    await page.locator('#chart-weekly .hit').first().hover();
    check(await page.locator('#chart-tip').isVisible(), '悬停提示');
    await shot(page, 'dashboard');
    steps.push('dashboard');

    // ---------------------------------------------------------------- 合规与部署：打码演示
    await page.getByRole('navigation', { name: '主导航' }).getByRole('link', { name: '合规与部署' }).click();
    await page.locator('#mask-run').click();
    await page.locator('#mask-output mark').first().waitFor();
    const masked = await page.locator('#mask-output').textContent();
    check(!masked.includes('13812345678') && masked.includes('[手机号]') && masked.includes('o_1002'), `打码：${masked}`);
    check((await page.locator('.unit').count()) === 3, '三个平台单元');
    await shot(page, 'compliance');
    steps.push('compliance');

    // ---------------------------------------------------------------- 切换平台
    await page.locator('.platform-btn[data-platform="douyin"]').click();
    check((await page.evaluate(() => document.body.dataset.platform)) === 'douyin', '抖店');
    check((await page.locator('.platform-btn[data-platform="douyin"]').getAttribute('aria-pressed')) === 'true', '按下状态');
    await page.getByRole('navigation', { name: '主导航' }).getByRole('link', { name: '智能辅助工作台' }).click();
    await page.locator('#mode-label').filter({ hasText: '飞鸽' }).waitFor();
    await page.locator('.platform-btn[data-platform="pdd"]').click();
    await page.locator('#mode-label').filter({ hasText: '模式 C' }).waitFor();
    check((await page.locator('.conv-btn[data-scenario="logistics"]').count()) === 0, '拼多多没有订单场景');
    await page.locator('.conv-btn[data-scenario="size"]').click();
    await page.locator('.sg-card').first().waitFor();
    check((await page.locator('.sg-card button[data-action="sent"]').first().textContent()) === '复制并标记已发送', '模式 C 按钮');
    steps.push('platform-switch');

    // 重置演示（只重置当前平台）
    await page.locator('#reset').click();
    await page.locator('#dialog-ok').click();
    await page.locator('#toast').filter({ hasText: '已重置' }).waitFor();
    await page.locator('.conv-btn[data-scenario="size"]').waitFor();
    check((await page.locator('.conv-btn[data-scenario="size"] .badge-ok').count()) === 0, '重置后会话清空');
    steps.push('reset');
    await cspClean(page, '桌面');

    // ---------------------------------------------------------------- 手机宽度
    const phone = await newPage(browser, { width: 390, height: 844 }, problems);
    for (const route of ['overview', 'workbench', 'aftersale', 'dashboard', 'compliance']) {
      await phone.goto(`${BASE}/#/${route}`);
      await phone.locator(`#page-${route}`).waitFor();
      if (route === 'overview') await phone.locator('#targets .kpi').first().waitFor();
      if (route === 'aftersale') await phone.locator('.as-card').first().waitFor();
      if (route === 'dashboard') await phone.locator('#chart-weekly svg').waitFor();
      if (route === 'workbench') {
        await phone.locator('.conv-btn[data-scenario="qa_block"]').click();
        await phone.locator('.sg-card').first().waitFor();
        await phone.locator('#trace-toggle').click();
        await phone.locator('#trace').waitFor();
        await shot(phone, 'phone-workbench');
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
