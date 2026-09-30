/*
 * 网页工具的登录状态和请求顺序（由 test_web_browser.py 启动；完整版 Chromium + Playwright）。
 *
 * - 往返缓存（bfcache）：Playwright 默认加 --disable-back-forward-cache，真实的 Chrome 默认开着，这里去掉
 *   这个参数。一个页面退出或换人登录后按「后退」，从缓存恢复的另一个页面不能还是原来的身份
 *   （window 上的标记证明确实是从缓存恢复的，不是重新加载）；
 * - 拒绝的角色（控制台里的客服、话术助手里的复核人员）页头有「退出」，退出清掉令牌和错误提示；
 * - 写操作还没回来时切到受影响的页签：回来后原地刷新，打开的修改面板里写了一半的内容不丢；
 * - 同一块区域晚回来的旧响应（换筛选、换商品）不覆盖新的；
 * - 上一位没完成的同步不挡下一位的按钮，晚回来时也不改下一位的按钮；
 * - 同步商品后，商家改过、还没保存的字段留在输入框里。
 * 慢请求用 page.route 拦住、手动放行（不靠计时）。
 *
 * 环境变量：RODIT_BASE_URL 测试服务器地址；RODIT_SMOKE_TOKENS 合成的测试令牌
 * {merchant, manager, agent, reviewer, merchant_b}。
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
const FULL_CHROMIUM = '/opt/pw-browsers/chromium';
// Playwright 默认关掉往返缓存；真实的 Chrome 默认开着
const BFCACHE = { ignoreDefaultArgs: ['--disable-back-forward-cache'] };

const steps = [];

function check(condition, message) {
  if (!condition) throw new Error(`检查失败：${message}`);
}

/** 完整版 Chromium（headless shell 没有往返缓存）。 */
async function launch() {
  const errors = [];
  const attempts = [{ channel: 'chromium', ...BFCACHE }];
  if (fs.existsSync(FULL_CHROMIUM)) attempts.push({ executablePath: FULL_CHROMIUM, ...BFCACHE });
  for (const options of attempts) {
    try {
      return await chromium.launch(options);
    } catch (err) {
      errors.push(err.message.split('\n')[0]);
    }
  }
  process.stderr.write(`Chromium 起不来：${errors.join(' | ')}\n`);
  process.exit(SKIP);
  return null;
}

async function main() {
  const browser = await launch();
  const problems = [];
  const urls = [];

  /** 新的标签页（新的 sessionStorage）。requests 记下每个接口请求用的令牌。 */
  async function newPage() {
    const context = await browser.newContext({ viewport: { width: 1000, height: 900 }, locale: 'zh-CN' });
    const page = await context.newPage();
    page.setDefaultTimeout(10000);
    page.on('console', (msg) => {
      const text = msg.text();
      if ((msg.type() === 'error' || msg.type() === 'warning') && !text.startsWith('Failed to load resource')) {
        problems.push(`console.${msg.type()}: ${text}`);
      }
    });
    page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`));
    const requests = [];
    page.on('request', (req) => {
      urls.push(req.url());
      if (!req.url().includes('/v1/')) return;
      const auth = req.headers().authorization || '';
      requests.push({ method: req.method(), path: new URL(req.url()).pathname, token: auth.replace(/^Bearer /, '') });
    });
    return { context, page, requests };
  }

  /** 拦住符合条件的请求，手动放行：seen 在请求到达时完成，release() 之后照常发出。 */
  async function hold(page, match) {
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    let arrived;
    const seen = new Promise((resolve) => {
      arrived = resolve;
    });
    await page.route(match, async (route) => {
      arrived(route.request());
      await gate;
      await route.continue();
    });
    return { seen, release: () => release() };
  }

  /** 放行后等响应到了浏览器，再给页面脚本一点时间处理（只用于"不应该变"的检查）。 */
  async function settle(page, held, match) {
    const response = page.waitForResponse(match);
    held.release();
    await response;
    await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 300)));
  }

  const storageOf = (page) => page.evaluate(() => Object.assign({}, sessionStorage));
  const tokensAfter = (requests, from) => requests.slice(from).map((r) => r.token);

  async function login(page, token) {
    await page.getByLabel('访问令牌').fill(token);
    await page.getByRole('button', { name: '登录', exact: true }).click();
  }

  // 每个场景用新的标签页（新的 sessionStorage），按顺序跑
  const SCENARIOS = [
    // ================================================================ 往返缓存：话术助手退出，后退回控制台
    [
      'bfcache-console-signed-out',
      async () => {
        const { context, page, requests } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        await page.locator('#faqs .faq').first().waitFor();
        await page.evaluate(() => {
          window.__marker = 'console-1';
        });
        await page.getByRole('link', { name: '话术助手（模式 C）' }).click();
        await page.locator('#identity').filter({ hasText: '商家' }).waitFor(); // 同一标签页：自动登录
        await page.getByRole('button', { name: '退出' }).click();
        await page.getByLabel('访问令牌').waitFor();
        check(Object.keys(await storageOf(page)).length === 0, '话术助手退出清掉 sessionStorage');
        const from = requests.length;
        await page.goBack({ waitUntil: 'commit' });
        check((await page.evaluate(() => window.__marker)) === 'console-1', '控制台确实是从往返缓存恢复的');
        await page.locator('#login').waitFor();
        await page.locator('#error').filter({ hasText: '已在另一个页面退出登录' }).waitFor();
        check(await page.locator('#app').isHidden(), '恢复后控制台不再可用');
        check(await page.locator('#whoami').isHidden(), '恢复后不显示上一位的身份');
        check((await page.locator('#faqs').textContent()) === '', '恢复后不留上一位的数据');
        // 就算有人绕过界面提交表单，也不会用内存里的旧令牌发请求
        await page.evaluate(() => document.getElementById('faq-form').requestSubmit());
        await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 300)));
        check(!tokensAfter(requests, from).includes(TOKENS.merchant), '后退后没有请求带着退出了的令牌');
        await context.close();
      },
    ],
    // ================================================================ 往返缓存：话术助手换人登录，后退回控制台
    [
      'bfcache-console-switched',
      async () => {
        const { context, page, requests } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        await page.locator('#faqs .faq').first().waitFor();
        await page.evaluate(() => {
          window.__marker = 'console-2';
        });
        await page.getByRole('link', { name: '话术助手（模式 C）' }).click();
        await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
        await page.getByRole('button', { name: '退出' }).click();
        await login(page, TOKENS.manager);
        await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
        const from = requests.length;
        await page.goBack({ waitUntil: 'commit' });
        check((await page.evaluate(() => window.__marker)) === 'console-2', '控制台确实是从往返缓存恢复的');
        await page.locator('#identity').filter({ hasText: '店长' }).waitFor();
        await page.locator('#faqs .faq').first().waitFor();
        check(await page.locator('#faq-create').isHidden(), '换成店长后没有商家的写按钮');
        const used = tokensAfter(requests, from);
        check(used.length > 0 && used.every((t) => t === TOKENS.manager), '后退后的请求都用店长的令牌');
        check((await storageOf(page))['rodit.assist.token'] === TOKENS.manager, '恢复时没有清掉店长的令牌');
        // 这时退出：清掉的是自己（店长）的令牌
        await page.getByRole('button', { name: '退出' }).click();
        await page.getByLabel('访问令牌').waitFor();
        check(Object.keys(await storageOf(page)).length === 0, '退出清掉 sessionStorage');
        await context.close();
      },
    ],
    // ================================================================ 往返缓存：控制台退出，后退回话术助手
    [
      'bfcache-assist-signed-out',
      async () => {
        const { context, page, requests } = await newPage();
        await page.goto(`${BASE}/assist`);
        await login(page, TOKENS.merchant);
        await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
        await page.locator('#adoption .headline, #adoption .muted').first().waitFor();
        await page.locator('#gaps li, #gaps .muted').first().waitFor();
        await page.evaluate(() => {
          window.__marker = 'assist-1';
        });
        await page.getByRole('link', { name: '商家控制台' }).click();
        await page.locator('#identity').filter({ hasText: '商家' }).waitFor();
        await page.locator('#faqs .faq').first().waitFor();
        await page.getByRole('button', { name: '退出' }).click();
        await page.getByLabel('访问令牌').waitFor();
        const from = requests.length;
        await page.goBack({ waitUntil: 'commit' });
        check((await page.evaluate(() => window.__marker)) === 'assist-1', '话术助手确实是从往返缓存恢复的');
        await page.locator('#login').waitFor();
        await page.locator('#error').filter({ hasText: '已在另一个页面退出登录' }).waitFor();
        check(await page.locator('#assist').isHidden(), '恢复后话术助手不再可用');
        check(await page.locator('#whoami').isHidden(), '恢复后不显示上一位的身份');
        await page.evaluate(() => {
          document.getElementById('question').value = '能开发票吗';
          document.getElementById('ask-form').requestSubmit();
        });
        await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 300)));
        check(tokensAfter(requests, from).length === 0, '后退后没有用旧令牌提问');
        await context.close();
      },
    ],
    // ================================================================ 拒绝的角色：页头有「退出」
    [
      'refused-role',
      async () => {
        const { context, page } = await newPage();
        await page.goto(`${BASE}/assist`);
        await login(page, TOKENS.reviewer);
        await page.locator('#error').filter({ hasText: '复核人员请使用商家控制台' }).waitFor();
        check(await page.locator('#assist').isHidden(), '话术助手页不给复核人员显示工具');
        check((await page.locator('#identity').textContent()).includes('复核人员'), '页头显示是谁');
        await page.getByRole('button', { name: '退出' }).click();
        await page.locator('#whoami').waitFor({ state: 'hidden' });
        check(Object.keys(await storageOf(page)).length === 0, '退出清掉复核人员的令牌');
        check(await page.locator('#error').isHidden(), '退出时清掉提示');
        // 控制台里的客服同理
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.agent);
        await page.locator('#error').filter({ hasText: '话术助手' }).waitFor();
        check((await page.locator('#identity').textContent()).includes('客服'), '控制台页头显示客服');
        await page.getByRole('button', { name: '退出' }).click();
        await page.locator('#whoami').waitFor({ state: 'hidden' });
        check(Object.keys(await storageOf(page)).length === 0, '退出清掉客服的令牌');
        check(await page.locator('#error').isHidden(), '控制台退出时清掉提示');
        await context.close();
      },
    ],
    // ================================================================ 写操作回来时已经切到受影响的页签
    [
      'stale-current-tab',
      async () => {
        const { context, page } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        const gaps = page.locator('#gaps');
        await gaps.locator('li').filter({ hasText: '商品 i_777 · 尺码建议' }).waitFor();

        // 商品页保存尺码建议，没回来时切到「知识」：回来后「要补的知识」马上更新
        await page.getByRole('tab', { name: '商品', exact: true }).click();
        await page.locator('#items tr', { hasText: '高腰阔腿牛仔裤' }).getByRole('button', { name: '查看 / 编辑' }).click();
        const detail = page.locator('#item-detail');
        await detail.getByLabel('尺码建议').fill('合成：版型偏大，建议拍小一码');
        const save = await hold(page, (url) => url.pathname.endsWith('/console/items/i_777'));
        await detail.getByRole('button', { name: '保存商家填写的字段' }).click();
        await save.seen;
        await page.getByRole('tab', { name: '知识', exact: true }).click();
        save.release();
        await gaps.filter({ hasText: '暂时没有要补的知识' }).waitFor();

        // 政策页清空「赠品」（活动结束），没回来时切到「知识」并打开 FAQ 的修改面板写了一半：
        // 回来后「要补的知识」列出赠品，面板和写了一半的内容还在
        await page.getByRole('tab', { name: '政策', exact: true }).click();
        const gift = page.locator('.policy').filter({ has: page.getByRole('heading', { name: '赠品', exact: true }) });
        await gift.getByRole('button', { name: '清空' }).click();
        const clear = await hold(page, (url) => url.pathname.endsWith('/console/policies/gift/clear'));
        await gift.getByRole('button', { name: '确认清空' }).click();
        await clear.seen;
        await page.getByRole('tab', { name: '知识', exact: true }).click();
        const faq = page.locator('#faqs .faq').filter({ hasText: '发什么快递' });
        await faq.getByRole('button', { name: '修改' }).click();
        const draft = '合成：写了一半的回答';
        await faq.getByLabel('标准回答').fill(draft);
        clear.release();
        await gaps.locator('li').filter({ hasText: '赠品' }).waitFor();
        check((await faq.getByLabel('标准回答').inputValue()) === draft, '打开的修改面板里写了一半的内容不丢');
        check(await faq.getByRole('button', { name: '保存修改' }).isVisible(), '修改面板还开着');
        await context.close();
      },
    ],
    // ================================================================ 晚回来的旧响应不覆盖新的
    [
      'latest-response',
      async () => {
        const { context, page } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        await page.locator('#faqs .faq').first().waitFor();
        // FAQ 筛选：先选「已下线」（慢），再选「待审核」
        const retiredMatch = (url) => url.pathname.endsWith('/console/faqs') && url.searchParams.get('status') === 'retired';
        const retired = await hold(page, (url) => retiredMatch(url));
        await page.locator('#faq-status').selectOption('retired');
        await retired.seen;
        await page.locator('#faq-status').selectOption('candidate');
        await page.locator('#faqs .faq.st-candidate').first().waitFor();
        await settle(page, retired, (resp) => retiredMatch(new URL(resp.url())));
        check((await page.locator('#faq-status').inputValue()) === 'candidate', '筛选框是后选的');
        check((await page.locator('#faqs .faq.st-candidate').count()) > 0, '列表还是后选的筛选结果');
        check(!(await page.locator('#faqs').textContent()).includes('没有符合条件的 FAQ'), '没有被前一次的结果覆盖');

        // 商品卡片：先点 i_555（慢），再点 i_777
        await page.getByRole('tab', { name: '商品', exact: true }).click();
        const detailMatch = (url) => url.pathname.endsWith('/console/items/i_555');
        const slow = await hold(page, (url) => detailMatch(url));
        await page.locator('#items tr', { hasText: 'i_555' }).getByRole('button', { name: '查看 / 编辑' }).click();
        await slow.seen;
        await page.locator('#items tr', { hasText: 'i_777' }).getByRole('button', { name: '查看 / 编辑' }).click();
        const detail = page.locator('#item-detail');
        await detail.getByRole('heading', { name: /i_777/ }).waitFor();
        await settle(page, slow, (resp) => detailMatch(new URL(resp.url())));
        check((await detail.locator('h3').textContent()).includes('i_777'), '商品卡片还是后点的那件');
        check((await detail.getAttribute('data-item-id')) === 'i_777', '卡片和记下的商品一致');
        await context.close();
      },
    ],
    // ================================================================ 上一位没完成的同步不挡下一位
    [
      'busy-sign-out',
      async () => {
        const { context, page, requests } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        await page.locator('#identity').filter({ hasText: '店铺 s_1001' }).waitFor();
        await page.getByRole('tab', { name: '商品', exact: true }).click();
        await page.locator('#items tr').first().waitFor();
        const syncOf = (shop) => (url) => url.pathname === `/v1/shops/${shop}/console/items/sync`;
        const heldA = await hold(page, syncOf('s_1001'));
        const heldB = await hold(page, syncOf('s_2002'));
        await page.locator('#sync-all').click();
        await heldA.seen;
        await page.getByRole('button', { name: '退出' }).click();
        await login(page, TOKENS.merchant_b);
        await page.locator('#identity').filter({ hasText: '店铺 s_2002' }).waitFor();
        await page.getByRole('tab', { name: '商品', exact: true }).click();
        await page.locator('#items tr, #items .muted').first().waitFor();
        check(await page.locator('#sync-all').isEnabled(), '下一位的「同步商品」可以点');
        await page.locator('#sync-all').click();
        const sentB = await heldB.seen;
        check(sentB.headers().authorization === `Bearer ${TOKENS.merchant_b}`, '下一位的同步用自己的令牌');
        check(await page.locator('#sync-all').isDisabled(), '下一位的同步进行中');
        // 上一位的同步这时才回来：不改下一位的按钮和报告
        await settle(page, heldA, (resp) => syncOf('s_1001')(new URL(resp.url())));
        check(await page.locator('#sync-all').isDisabled(), '上一位的同步回来不把下一位的按钮放开');
        check(!(await page.locator('#sync-report').textContent()).includes('s_1001'), '上一位的同步结果不显示');
        heldB.release();
        await page.locator('#sync-report').filter({ hasText: '同步完成（全店）' }).waitFor();
        await page.waitForFunction(() => !document.getElementById('sync-all').disabled);
        const syncs = requests.filter((r) => r.path.endsWith('/console/items/sync'));
        check(syncs.length === 2 && syncs[1].path.includes('s_2002'), '下一位的点击真的发出了请求');
        await context.close();
      },
    ],
    // ================================================================ 同步后保留还没保存的商家字段
    [
      'sync-keeps-drafts',
      async () => {
        const { context, page, requests } = await newPage();
        await page.goto(`${BASE}/console`);
        await login(page, TOKENS.merchant);
        await page.getByRole('tab', { name: '商品', exact: true }).click();
        await page.locator('#items tr', { hasText: 'i_555' }).getByRole('button', { name: '查看 / 编辑' }).click();
        const detail = page.locator('#item-detail');
        await detail.getByRole('heading', { name: /i_555/ }).waitFor();
        const draft = '合成：草稿，还没保存';
        await detail.getByLabel('尺码建议').fill(draft);
        const care = await detail.getByLabel('洗涤保养').inputValue();
        await detail.getByRole('button', { name: '同步这件商品' }).click();
        await page.locator('#sync-report').filter({ hasText: '同步完成（商品 i_555）' }).waitFor();
        const note = detail.locator('.item-form .status.warn');
        await note.filter({ hasText: '还没保存' }).waitFor();
        check((await note.textContent()).includes('尺码建议'), '提示哪些字段还没保存');
        check((await detail.getByLabel('尺码建议').inputValue()) === draft, '同步后草稿还在');
        check((await detail.getByLabel('洗涤保养').inputValue()) === care, '没改的字段是服务端的内容');
        check(!requests.some((r) => r.method === 'POST' && r.path.endsWith('/console/items/i_555')), '同步不替商家保存');
        // 全店同步同样保留（给旧表单做个记号，检查的是同步后重新显示的那张）
        await detail.locator('.item-form').evaluate((form) => {
          form.dataset.before = 'sync-all';
        });
        await page.locator('#sync-all').click();
        await page.locator('#sync-report').filter({ hasText: '同步完成（全店）' }).waitFor();
        const fresh = detail.locator('.item-form:not([data-before])');
        await fresh.locator('.status.warn').filter({ hasText: '还没保存' }).waitFor();
        check((await fresh.getByLabel('尺码建议').inputValue()) === draft, '全店同步后草稿还在');
        await context.close();
      },
    ],
  ];

  try {
    for (const [name, run] of SCENARIOS) {
      await run();
      steps.push(name);
    }

    // ---------------------------------------------------------------- 令牌从不出现在网址里
    const secrets = Object.values(TOKENS);
    check(!urls.some((u) => secrets.some((t) => u.includes(t))), '令牌不进网址');
    check(problems.length === 0, `浏览器报错：${problems.join(' | ')}`);
    process.stdout.write(`${JSON.stringify({ ok: true, steps, requests: urls.length, browser: browser.version() })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  process.stdout.write(`${JSON.stringify({ ok: false, steps, error: String(err && err.stack ? err.stack : err) })}\n`);
  process.exit(1);
});
