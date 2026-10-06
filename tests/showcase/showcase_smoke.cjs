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
    await page.locator('#mode-label').filter({ hasText: '千牛' }).waitFor(); // 等 /scenarios 返回（第一次要组装单元）
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
    const simulatedText = await page.locator('#ai-simulated').textContent();
    check(simulatedText.includes('演示：模拟大模型先写出没有依据的承诺') && simulatedText.includes('下面是重写后通过的回答'), `模拟说明：${simulatedText}`);
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

    // ---------------------------------------------------------------- 半自动 + 上线节奏
    check(!(await page.locator('#rollout').isHidden()), '上线节奏');
    check((await page.locator('#rollout .rollout-step').count()) === 4, '上线节奏四步');
    check((await page.locator('#rollout .rollout-step[data-step="assist"]').getAttribute('aria-current')) === 'step', '当前是智能辅助');
    const rolloutText = await page.locator('#rollout').textContent();
    check(rolloutText.includes('发货与快递') && rolloutText.includes('强制开启'), `已开半自动的类型：${rolloutText}`);
    check(rolloutText.includes('09:00-23:00') && rolloutText.includes('工作时间以外试行'), '全自动先在非工作时间试行');
    await page.locator('.conv-btn[data-scenario="ship_quick"]').click();
    const quick = page.locator('.sg-card[data-prefill="true"]').first();
    await quick.waitFor();
    check((await quick.locator('.badge-prefill').textContent()) === '半自动：已填入输入框，按发送即可', '半自动徽标');
    check((await quick.locator('.prefill-banner').getAttribute('data-prefill')) === 'input', '填进输入框');
    await shot(page, 'workbench-prefill');
    await quick.locator('button[data-action="sent"]').click();
    await page.locator('.msg-agent .bubble').filter({ hasText: '48 小时' }).waitFor();
    check((await page.locator('.sg-card .sg-done').first().textContent()).includes('已原样发送'), '半自动仍由客服发送');
    steps.push('prefill');

    // ---------------------------------------------------------------- 买家图片（§15 Phase 3）
    await page.locator('.conv-btn[data-scenario="photo_damage"]').click();
    await page.locator('.msg-buyer .img-tile[data-ref="img_buyer_damage_1"]').waitFor();
    const photo = page.locator('.sg-card').first();
    await photo.waitFor();
    check((await photo.locator('textarea').inputValue()).includes('您发来的照片我们看到了'), '照片模板');
    check((await page.locator('#ai-alerts .alert-image').count()) === 1, '图片摘要提醒');
    await page.locator('#trace .trace-images').waitFor();
    const imagesText = await page.locator('#trace .trace-images').textContent();
    check(imagesText.includes('破损') && imagesText.includes('93%') && imagesText.includes('【买家发来图片：破损'), `识别结果：${imagesText}`);
    await shot(page, 'workbench-image');
    await page.locator('.conv-btn[data-scenario="photo_logistics"]').click();
    await page.locator('#trace .finding-ocr-text').waitFor();
    check((await page.locator('#trace .finding-ocr-text').textContent()).includes('[手机号]'), '图中文字已打码');
    check(!(await page.locator('#page-workbench').textContent()).includes('13812345678'), '页面上没有手机号原文');
    // 自己选图发：在问面料和发货的会话里（商品卡片是那条连衣裙），只发一张身高体重截图
    await page.locator('.conv-btn[data-scenario="ship_time"]').click();
    await page.locator('.msg-buyer .bubble').filter({ hasText: '多久能发货' }).waitFor();
    await page.locator('#composer-image').click();
    await page.locator('#image-picker').waitFor();
    check((await page.locator('#image-picker .picker-tile').count()) === 6, '可选的合成图片');
    await page.locator('.picker-tile[data-ref="img_buyer_size_1"]').click();
    check((await page.locator('.picker-tile[data-ref="img_buyer_size_1"]').getAttribute('aria-pressed')) === 'true', '选中图片');
    await page.locator('#composer-images .img-tile[data-ref="img_buyer_size_1"]').waitFor();
    await page.locator('#composer-send').click();
    await page.locator('.msg-buyer .img-tile[data-ref="img_buyer_size_1"]').waitFor();
    const sized = page.locator('.sg-card[data-status="shown"]').first();
    await sized.waitFor();
    check((await sized.locator('textarea').inputValue()).includes('M'), '截图里的身高体重 → M 码');
    check((await page.locator('#composer-images .img-tile').count()) === 0, '发完清空待发图片');
    check(await page.locator('#image-picker').isHidden(), '发完收起选择框');
    steps.push('image');

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
    // R5 质量问题单：凭证图片理解的摘要（只是参考）
    const vision = page.locator('.as-card[data-id="as_2009"] .as-vision');
    check((await vision.textContent()).includes('3 张凭证中 2 张显示破损'), '凭证图片理解');
    check((await vision.textContent()).includes('凭证图片 3 张'), '凭证张数');
    check((await vision.locator('.img-tile').count()) === 3, '凭证占位图块');
    check((await page.locator('#aftersale-desk-note').textContent()).includes('/aftersale'), '生产工作台说明');
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
    check((await page.locator('#chart-metrics .verdict').textContent()).includes('较接入前改善'), '总体结论');
    check((await page.locator('#chart-metrics .weekly-bars .wbar').count()) === 8, '按周序列（前后各 4 周）');
    await page.locator('#chart-metrics .weekly-bars .wbar.is-after').first().hover();
    check((await page.locator('#chart-tip').textContent()).includes('接入后'), '按周的悬停提示');
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
    check(!masked.includes('13912345678') && masked.includes('[手机号]') && masked.includes('o_1002'), `打码：${masked}`);
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
    check(await page.locator('#composer-image').isDisabled(), '模式 C 不能发图片');
    check((await page.locator('#composer-note').textContent()).includes('模式 C'), '说明为什么不能发图片');
    check((await page.locator('.conv-btn[data-scenario="photo_damage"]').count()) === 0, '拼多多没有图片场景');
    await page.locator('.conv-btn[data-scenario="invoice"]').click();
    await page.locator('.sg-card[data-prefill="true"] .badge-prefill').filter({ hasText: '可直接发送' }).waitFor();
    steps.push('platform-switch');

    // 页面载入后演示被重置过（别的窗口、服务重启）：旧卡片上的操作带着旧 epoch，服务端 409，
    // 页面提示并重新载入，不会落到重置后同编号的推荐上
    const reset = await page.request.post(`${BASE}/showcase/api/reset`, { data: { platform: 'pdd' } });
    check(reset.ok(), '接口重置');
    const known = problems.length;
    const stale = page.locator('.sg-card[data-status="shown"]').first();
    await stale.locator('button[data-action="rejected"]').click();
    await stale.locator('button[data-reason="other"]').click();
    await page.locator('#toast').filter({ hasText: '这次操作没有执行' }).waitFor();
    await page.waitForFunction(() => !document.querySelector('.conv-btn .badge-ok')); // 会话列表重新载入
    check((await page.locator('.sg-card').count()) === 0, '重新载入后没有旧卡片');
    // 浏览器把 409 响应记成一条控制台错误：这一步预期之内，其他问题照常算
    problems.splice(known).filter((p) => !p.includes('409')).forEach((p) => problems.push(p));
    steps.push('stale-epoch');

    // 重置演示（只重置当前平台）
    await page.locator('#reset').click();
    await page.locator('#dialog-ok').click();
    await page.locator('#toast').filter({ hasText: '回到初始状态' }).waitFor();
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
        await noHorizontalScroll(phone, '手机 工作台');
        await phone.locator('.conv-btn[data-scenario="photo_size"]').click();
        await phone.locator('#trace .trace-images').waitFor();
        await phone.locator('#composer-image').click();
        await phone.locator('#image-picker').waitFor();
        await shot(phone, 'phone-workbench-image');
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
