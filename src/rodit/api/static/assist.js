/*
 * Rodit 模式 C 独立话术助手（方案 §8.1 模式 C、§5.6 推荐卡片、§5.8 反馈、§12.4 商家报表）。
 *
 * 约定：
 * - 服务端和用户给的文字一律用 textContent / 文本节点写进页面，从不按 HTML 解析（防 XSS）；
 * - 令牌只放 sessionStorage，只放进 Authorization 请求头，不进网址、不打日志；
 * - 接口地址用相对路径（页面在 / 和 /assist 下都能用）；
 * - 错误码和说明来自接口的统一错误格式 {"error", "message"}（rodit.api.errors）；
 * - 和商家控制台共用 sessionStorage 里的令牌：退出、令牌无效时只在存着的还是本页的令牌时才清
 *   （forgetToken）；从往返缓存（bfcache）恢复时存着的令牌变了，先丢掉本页内存里的身份再重新登录；
 * - 同一块区域的加载按请求编号只用最近一次的结果（nextRun / latest）。
 */
'use strict';

(() => {
  const $ = (id) => document.getElementById(id);

  const CONFIG = (() => {
    const node = $('rodit-config');
    try {
      return JSON.parse(node ? node.textContent : '{}') || {};
    } catch (err) {
      return {};
    }
  })();

  const LABELS = {
    intents: CONFIG.intents || {},
    rejectReasons: CONFIG.reject_reasons || {},
    confidence: CONFIG.confidence || {},
    kinds: CONFIG.kinds || {},
    roles: CONFIG.roles || {},
    degraded: CONFIG.degraded || {},
  };
  const MAX_TEXT = CONFIG.max_text_chars || 2000;
  const HUMAN_ROLES = CONFIG.human_roles || ['agent', 'manager', 'merchant'];
  const CONSOLE_ROLES = CONFIG.console_roles || ['manager', 'merchant'];
  const MERCHANT_ROLES = CONFIG.merchant_roles || ['merchant'];
  const REVIEWER_ROLES = CONFIG.reviewer_roles || ['reviewer'];
  const ADOPTION_TARGET = typeof CONFIG.adoption_target === 'number' ? CONFIG.adoption_target : 0.5;
  const ITEM_FILTER_MIN = 12; // 商品多于这个数时显示筛选框
  const MISSING_ITEM = 'item:?';

  const STORAGE_TOKEN = 'rodit.assist.token';
  const STORAGE_SESSION = 'rodit.assist.session';
  const STORAGE_TAB = 'rodit.console.tab'; // 商家控制台的页签偏好：退出时一并清掉

  const state = {
    token: null,
    stored: null, // 本页写进 sessionStorage 的令牌（退出时只清自己的；存不了时为 null）
    me: null,
    items: [],
    sessionId: null,
    asked: 0,
    cards: [], // 本会话里所有推荐卡片的状态（新问题提交成功后，还没操作的都过期）
    busy: false,
  };

  // ================================================================ 小工具

  /** 建元素。props.text 用 textContent；children 里的字符串是文本节点。不接受 HTML 字符串。 */
  function h(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [key, value] of Object.entries(props)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'text') node.textContent = String(value);
        else if (key === 'className') node.className = value;
        else if (key === 'hidden') node.hidden = true;
        else if (key.startsWith('on') && typeof value === 'function') {
          node.addEventListener(key.slice(2), value);
        } else node.setAttribute(key, value === true ? '' : String(value));
      }
    }
    for (const child of [].concat(children === undefined ? [] : children)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const storage = {
    get(key) {
      try {
        return window.sessionStorage.getItem(key);
      } catch (err) {
        return null;
      }
    },
    set(key, value) {
      try {
        window.sessionStorage.setItem(key, value);
      } catch (err) {
        /* 存不了（隐私模式等）：只放在内存里，刷新后要重新登录 */
      }
    },
    remove(key) {
      try {
        window.sessionStorage.removeItem(key);
      } catch (err) {
        /* 同上 */
      }
    },
  };

  /** 把登录用的令牌存进 sessionStorage，记下是本页存的（存不了时不记：两个页面本来也共用不了）。 */
  function keepToken(token) {
    storage.set(STORAGE_TOKEN, token);
    state.stored = storage.get(STORAGE_TOKEN) === token ? token : null;
  }

  /**
   * 清掉 sessionStorage 里的令牌、会话和控制台的页签偏好——只在存着的还是这个令牌时才清：
   * 另一个页面后来登录的人（同一标签页里后退、前进回来时）不受本页退出或 401 的影响。
   */
  function forgetToken(token) {
    if (!token || storage.get(STORAGE_TOKEN) !== token) return;
    storage.remove(STORAGE_TOKEN);
    storage.remove(STORAGE_SESSION);
    storage.remove(STORAGE_TAB);
  }

  // 请求顺序：同一块区域只用最近一次请求的结果（晚回来的旧响应丢掉）
  const runs = {};

  function nextRun(key) {
    runs[key] = (runs[key] || 0) + 1;
    return runs[key];
  }

  const latest = (key, run) => runs[key] === run;

  /** 作废所有区域还没回来的加载（退出时）。 */
  function dropRuns() {
    for (const key of Object.keys(runs)) nextRun(key);
  }

  function randomId(prefix) {
    const bytes = new Uint8Array(8);
    window.crypto.getRandomValues(bytes);
    return `${prefix}-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
  }

  const enc = encodeURIComponent;

  function pct(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    return `${Math.round(value * 1000) / 10}%`;
  }

  function label(map, key) {
    return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : String(key);
  }

  function degradedLabel(code) {
    const text = String(code);
    if (Object.prototype.hasOwnProperty.call(LABELS.degraded, text)) return LABELS.degraded[text];
    const [kind, ...rest] = text.split(':');
    if (kind === 'tool' && LABELS.degraded.tool) return `${LABELS.degraded.tool}（${rest.join(':')}）`;
    return text;
  }

  let liveTimer = null;
  function announce(message) {
    const live = $('live');
    live.textContent = '';
    window.clearTimeout(liveTimer);
    liveTimer = window.setTimeout(() => {
      live.textContent = message;
    }, 50);
  }

  /** 页面顶部的错误区。link 可选：{href, text}，比如去商家控制台。 */
  function showError(message, link) {
    const box = $('error');
    box.textContent = message;
    if (link) box.append(' ', h('a', { href: link.href }, link.text));
    box.hidden = false;
  }

  function clearError() {
    const box = $('error');
    box.textContent = '';
    box.hidden = true;
  }

  // ================================================================ 接口

  class ApiFailure extends Error {
    constructor(status, code, message) {
      super(message);
      this.status = status;
      this.code = code;
    }
  }

  const STATUS_TEXT = {
    0: '网络错误',
    401: '未登录或令牌无效',
    403: '没有权限',
    404: '找不到',
    409: '操作冲突',
    422: '输入有误',
    500: '服务出错',
    503: '服务暂时不可用',
  };

  function describeFailure(status, message) {
    const title = STATUS_TEXT[status] || (status >= 500 ? '服务出错' : '请求失败');
    const where = status ? `（${status}）` : '';
    const text = typeof message === 'string' && message.trim() ? message.trim() : '请稍后重试';
    return `${title}${where}：${text}`;
  }

  async function api(method, path, body) {
    const headers = { Accept: 'application/json' };
    const token = state.token; // 发请求时的身份：响应回来时可能已经换人登录
    if (token) headers.Authorization = `Bearer ${token}`;
    const init = {
      method,
      headers,
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
    };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    let resp;
    try {
      resp = await fetch(path, init);
    } catch (err) {
      throw new ApiFailure(0, 'network', describeFailure(0, '连不上服务器，请检查网络后重试'));
    }
    if (resp.status === 204) return null;
    let data = null;
    try {
      data = await resp.json();
    } catch (err) {
      data = null;
    }
    if (!resp.ok) {
      const code = data && typeof data.error === 'string' ? data.error : `http_${resp.status}`;
      const failure = new ApiFailure(resp.status, code, describeFailure(resp.status, data && data.message));
      if (resp.status === 401 && state.me && state.token === token) {
        signOut();
        showError(`${failure.message}。请重新登录。`);
      }
      throw failure;
    }
    return data;
  }

  const shopPath = (suffix) => `v1/shops/${enc(state.me.shop_id)}/${suffix}`;

  // ================================================================ 登录

  let signIns = 0; // 第几次登录：退出、后退恢复时加一，还没回来的登录作废

  async function signIn(token) {
    const run = (signIns += 1);
    clearError();
    state.token = token;
    let me;
    try {
      me = await api('GET', 'v1/me');
    } catch (err) {
      if (run !== signIns) return false; // 期间又登录、退出或后退恢复时换了令牌：这次登录作废
      state.token = null;
      // 令牌无效才丢掉；网络、服务故障时留着，刷新后还能直接登录
      if (err.status === 401 || err.status === 403) forgetToken(token);
      if (err.code === 'forbidden_role') {
        showError('这个令牌是平台对接用的，不能登录网页工具；请用客服、店长或商家的令牌。');
      } else {
        showError(err.message);
      }
      $('token').focus();
      return false;
    }
    if (run !== signIns) return false;
    if (me && REVIEWER_ROLES.includes(me.role)) {
      // 复核人员的工作（每周复核、上线节奏）在商家控制台，这里不显示话术助手。
      // 令牌留着（两个页面同一个键），同一标签页打开控制台时直接登录；
      // 页头显示是谁、给「退出」按钮，不想交给控制台时可以丢掉
      state.token = null;
      keepToken(token);
      $('token').value = '';
      showIdentity(me);
      showError('复核人员请使用商家控制台（每周复核、上线节奏），话术助手只给客服、店长和商家用。', {
        href: 'console',
        text: '打开商家控制台',
      });
      return false;
    }
    if (!me || !HUMAN_ROLES.includes(me.role)) {
      state.token = null;
      forgetToken(token);
      showError('这个令牌不能登录网页工具。');
      return false;
    }
    state.me = me;
    keepToken(token);
    enterApp();
    return true;
  }

  function showIdentity(me) {
    $('whoami').hidden = false;
    $('identity').textContent = `${label(LABELS.roles, me.role)} ${me.user_id} · 店铺 ${me.shop_id}`;
  }

  function enterApp() {
    const me = state.me;
    $('token').value = '';
    $('notice').open = false; // 登录后收起须知，摘要里的协议要求一直可见
    $('login').hidden = true;
    $('assist').hidden = false;
    showIdentity(me);
    restoreSession();
    loadItems();
    const isConsole = CONSOLE_ROLES.includes(me.role);
    $('console').hidden = !isConsole;
    $('questionnaire').hidden = !MERCHANT_ROLES.includes(me.role);
    if (isConsole) loadConsole();
    $('question').focus();
    announce(`已登录：${$('identity').textContent}`);
  }

  /**
   * 退出：清掉本页的身份、会话记录和表单。sessionStorage 只在存着的还是本页的令牌时才清（forgetToken）；
   * opts.keepStorage：只清本页（后退恢复时发现另一个页面已经退出或换人登录，存着的令牌归那边）。
   */
  function signOut(opts = {}) {
    if (!opts.keepStorage) forgetToken(state.stored);
    signIns += 1; // 还没回来的登录作废
    state.token = null;
    state.stored = null;
    state.me = null;
    state.items = [];
    state.cards = [];
    state.sessionId = null;
    state.asked = 0;
    dropRuns();
    setBusy(false); // 上一位还没回来的提问不挡下一位
    $('questionnaire-button').disabled = false;
    $('turns').replaceChildren();
    $('question').value = '';
    $('item-filter').value = '';
    renderItemOptions();
    $('adoption').replaceChildren();
    $('gaps').replaceChildren();
    $('questionnaire-status').textContent = '';
    for (const area of $('questionnaire-fields').querySelectorAll('textarea')) area.value = '';
    $('assist').hidden = true;
    $('console').hidden = true;
    $('whoami').hidden = true;
    $('identity').textContent = '';
    $('notice').open = true;
    $('login').hidden = false;
    $('token').focus();
  }

  // ================================================================ 会话（§5.3：同一会话的追问带上前面的对话）

  function saveSession() {
    const me = state.me;
    storage.set(
      STORAGE_SESSION,
      JSON.stringify({ shop_id: me.shop_id, user_id: me.user_id, session_id: state.sessionId }),
    );
  }

  function restoreSession() {
    let saved = null;
    try {
      saved = JSON.parse(storage.get(STORAGE_SESSION) || 'null');
    } catch (err) {
      saved = null;
    }
    const me = state.me;
    if (
      saved &&
      saved.shop_id === me.shop_id &&
      saved.user_id === me.user_id &&
      typeof saved.session_id === 'string'
    ) {
      state.sessionId = saved.session_id;
      state.asked = 0;
      state.cards = [];
      $('turns').replaceChildren();
      updateSessionLabel();
    } else {
      newSession();
    }
  }

  function newSession() {
    state.sessionId = randomId('web');
    state.asked = 0;
    state.cards = [];
    $('turns').replaceChildren();
    $('question').value = '';
    $('item').value = '';
    saveSession();
    updateSessionLabel();
  }

  function updateSessionLabel() {
    const suffix = state.asked ? ` · 已问 ${state.asked} 次` : '';
    $('session-label').textContent = `会话 ${state.sessionId}${suffix}`;
  }

  // ================================================================ 商品

  async function loadItems() {
    const hint = $('item-hint');
    const me = state.me; // 每次登录换一个对象：响应回来时已经退出或换人登录就丢掉
    const run = nextRun('items');
    try {
      const items = await api('GET', shopPath('items'));
      if (state.me !== me || !latest('items', run)) return;
      state.items = Array.isArray(items) ? items : [];
      hint.textContent = '买家问的是某件商品（面料、尺码……）时选上，回答更准。';
    } catch (err) {
      if (state.me !== me || !latest('items', run)) return;
      state.items = [];
      hint.textContent = `商品列表没有加载出来：${err.message}`;
    }
    $('item-filter-wrap').hidden = state.items.length <= ITEM_FILTER_MIN;
    renderItemOptions();
  }

  function itemLabel(item) {
    return `${item.title}（${item.item_id}）`;
  }

  function renderItemOptions() {
    const select = $('item');
    const current = select.value;
    const filter = $('item-filter').value.trim().toLowerCase();
    const options = [h('option', { value: '' }, '不指定商品')];
    for (const item of state.items) {
      const text = itemLabel(item);
      if (filter && item.item_id !== current && !text.toLowerCase().includes(filter)) continue;
      options.push(h('option', { value: item.item_id }, text));
    }
    select.replaceChildren(...options);
    select.value = state.items.some((i) => i.item_id === current) ? current : '';
  }

  function itemTitle(itemId) {
    const item = state.items.find((i) => i.item_id === itemId);
    return item ? itemLabel(item) : itemId;
  }

  // ================================================================ 提问（POST .../standalone/ask）

  function setBusy(busy) {
    state.busy = busy;
    const button = $('ask-button');
    button.disabled = busy;
    button.textContent = busy ? '正在生成推荐…' : '获取推荐回答';
    $('turns').setAttribute('aria-busy', busy ? 'true' : 'false');
  }

  async function ask() {
    if (state.busy || !state.me) return;
    clearError();
    const field = $('question');
    const question = field.value.trim();
    if (!question) {
      showError('请先粘贴买家的问题。');
      field.focus();
      return;
    }
    if (question.length > MAX_TEXT) {
      showError(`买家的问题太长了（最多 ${MAX_TEXT} 个字），请只粘贴问题本身。`);
      field.focus();
      return;
    }
    const itemId = $('item').value || null;
    const body = { session_id: state.sessionId, question, request_id: randomId('req') };
    if (itemId) body.item_id = itemId;

    const turn = addTurn(question, itemId);
    const me = state.me;
    setBusy(true);
    try {
      const result = await api('POST', shopPath('standalone/ask'), body);
      if (state.me !== me) return; // 期间退出或换人登录：不把上一位的结果写进新会话
      field.value = '';
      if (result === null) {
        turn.body.replaceChildren(h('p', { className: 'note' }, '这条问题没有处理（重复提交或不是买家问题）。'));
        announce('没有处理');
      } else {
        state.asked += 1;
        updateSessionLabel();
        expireCards();
        renderResult(turn, result);
      }
    } catch (err) {
      if (state.me !== me) return;
      turn.article.classList.add('failed');
      turn.body.replaceChildren(
        h('p', { className: 'error-inline', role: 'alert' }, `没有提交成功：${err.message}`),
      );
    } finally {
      if (state.me === me) setBusy(false); // 期间退出过：退出时已经复位，不动下一位的按钮
    }
  }

  function addTurn(question, itemId) {
    const body = h('div', { className: 'turn-body' }, h('p', { className: 'note' }, '正在生成推荐…'));
    const head = h('p', { className: 'turn-head' }, [
      h('span', { className: 'turn-label' }, '买家问'),
      itemId ? h('span', { className: 'tag' }, `商品：${itemTitle(itemId)}`) : null,
    ]);
    const article = h('article', { className: 'turn' }, [
      head,
      h('blockquote', { className: 'buyer' }, question),
      body,
    ]);
    $('turns').prepend(article);
    return { article, body };
  }

  function renderResult(turn, view) {
    const body = turn.body;
    body.replaceChildren();
    if (view.shadow) {
      body.append(
        h(
          'p',
          { className: 'note shadow' },
          '本店处于影子模式（§13.4）：推荐只在后台生成和记录，不在这里展示。请按平时的方式回复买家。',
        ),
      );
      announce('影子模式：不展示推荐');
      return;
    }
    const alerts = Array.isArray(view.alerts) ? view.alerts : [];
    if (view.transfer_to_human) {
      body.append(h('p', { className: 'alert' }, '建议人工优先处理这位买家。'));
    }
    if (alerts.length) {
      body.append(h('ul', { className: 'alerts', 'aria-label': '提醒' }, alerts.map((a) => h('li', null, a))));
    }
    const cards = Array.isArray(view.suggestions) ? view.suggestions : [];
    cards.forEach((card, index) => body.append(renderCard(card, index, cards.length)));
    if (view.no_suggestion_reason) {
      body.append(h('p', { className: 'no-suggestion' }, `没有推荐：${view.no_suggestion_reason}`));
    } else if (!cards.length) {
      body.append(h('p', { className: 'no-suggestion' }, '没有推荐。'));
    }
    const missing = Array.isArray(view.missing_knowledge_details) ? view.missing_knowledge_details : [];
    if (missing.length) {
      const items = missing.map((m) => h('li', null, m.description || m.marker));
      const box = h('div', { className: 'missing' }, [
        h('p', null, '缺少这些知识（商家补充后才能回答得更好）：'),
        h('ul', null, items),
      ]);
      if (missing.some((m) => m.marker === MISSING_ITEM)) {
        box.append(h('p', { className: 'hint' }, '可以在上面「商品」里选中买家问的商品，再问一次。'));
      }
      body.append(box);
    }
    const degraded = Array.isArray(view.degraded) ? view.degraded : [];
    if (degraded.length) {
      body.append(h('p', { className: 'degraded' }, `部分能力降级：${degraded.map(degradedLabel).join('、')}`));
    }
    announce(cards.length ? `已生成 ${cards.length} 条推荐` : '没有推荐');
  }

  // ================================================================ 推荐卡片（§5.6）与反馈（§5.8）

  function renderCard(card, index, total) {
    const titleId = randomId('card');
    const text = h('p', { className: 'card-text' }, card.text);
    const evidence = Array.isArray(card.evidence) ? card.evidence : [];
    const risks = Array.isArray(card.risk_hints) ? card.risk_hints : [];
    const cs = {
      card,
      status: 'shown',
      busy: false,
      manualCopy: false,
      text,
      actions: h('div', { className: 'actions' }),
      panel: h('div', { className: 'card-panel' }),
      statusLine: h('p', { className: 'status', role: 'status', tabindex: '-1' }),
      article: null,
      buttons: [],
    };
    const meta = h('p', { className: 'card-meta' }, [
      h('span', { className: `badge conf-${card.confidence}` }, `置信度 ${label(LABELS.confidence, card.confidence)}`),
      h('span', { className: 'badge' }, label(LABELS.intents, card.intent)),
      h('span', { className: 'badge' }, label(LABELS.kinds, card.kind)),
    ]);
    const evidenceBox = h('div', { className: 'evidence' }, [
      h('span', { className: 'evidence-title' }, '依据：'),
      evidence.length
        ? h('ul', { className: 'chips' }, evidence.map((e) => h('li', { className: 'chip' }, e.label)))
        : h('span', { className: 'muted' }, '无'),
    ]);
    const riskBox = risks.length
      ? h('ul', { className: 'risks', 'aria-label': '风险提示' }, risks.map((r) => h('li', null, r)))
      : null;
    cs.article = h('article', { className: 'card', 'aria-labelledby': titleId }, [
      h('h3', { id: titleId, className: 'card-title' }, total > 1 ? `推荐 ${index + 1} / ${total}` : '推荐回答'),
      meta,
      text,
      evidenceBox,
      riskBox,
      cs.actions,
      cs.panel,
      cs.statusLine,
    ]);

    if (typeof card.suggestion_id === 'number') {
      cs.sendButton = h('button', { type: 'button', onclick: () => onSend(cs) }, '复制并标记已发送');
      cs.editButton = h('button', { type: 'button', className: 'secondary', onclick: () => openEdit(cs) }, '编辑后发送');
      cs.rejectButton = h('button', { type: 'button', className: 'secondary', onclick: () => openReject(cs) }, '不合适');
      cs.buttons = [cs.sendButton, cs.editButton, cs.rejectButton];
      cs.actions.append(...cs.buttons);
      state.cards.push(cs);
    } else {
      // 存储降级时推荐没有编号（§12.2）：只能复制，记不了反馈
      const copy = h('button', { type: 'button', className: 'secondary' }, '复制');
      copy.addEventListener('click', async () => {
        const ok = await copyText(card.text);
        if (!ok) selectText(cs.text);
        setStatus(cs, ok ? '已复制。' : '浏览器不允许自动复制：文字已选中，请手动复制。', ok ? 'ok' : 'warn');
      });
      cs.actions.append(copy);
      setStatus(cs, '这条推荐没有记录编号（存储暂时不可用），无法记录反馈。', 'warn');
    }
    return cs.article;
  }

  function setStatus(cs, message, kind) {
    cs.statusLine.textContent = message;
    cs.statusLine.className = `status ${kind || ''}`.trim();
  }

  function setCardBusy(cs, busy) {
    cs.busy = busy;
    for (const button of cs.buttons) button.disabled = busy;
    for (const button of cs.panel.querySelectorAll('button')) button.disabled = busy;
  }

  function feedbackPath(cs, action) {
    return shopPath(`suggestions/${enc(String(cs.card.suggestion_id))}/${action}`);
  }

  /** 推荐已经处理完（发送、编辑、不合适、过期、冲突）：收起按钮，只留状态。 */
  function finishCard(cs, status, message, kind) {
    cs.status = status;
    cs.actions.replaceChildren();
    cs.panel.replaceChildren();
    cs.buttons = [];
    cs.article.classList.add('done', `status-${status}`);
    setStatus(cs, message, kind || 'ok');
    announce(message);
    cs.statusLine.focus();
  }

  /** 这张卡片还没处理完时，接口失败后的处理：409 说明推荐状态已经变了（过期或别处处理过），不能再操作。 */
  function failCard(cs, err, prefix) {
    if (err.status === 409 || err.status === 404) {
      finishCard(cs, 'conflict', `${prefix}${err.message}`, 'error');
    } else {
      setStatus(cs, `${prefix}${err.message}`, 'error');
      cs.statusLine.focus();
    }
  }

  /** 新问题提交成功：同一会话里还没处理的推荐都被服务端刷新成过期（§5.3），这里同步显示。 */
  function expireCards() {
    for (const cs of state.cards) {
      if (cs.status !== 'shown') continue;
      cs.status = 'expired';
      cs.actions.replaceChildren();
      cs.panel.replaceChildren();
      cs.buttons = [];
      cs.article.classList.add('done', 'status-expired');
      setStatus(cs, '买家有了新问题，这条推荐已过期，不再记录反馈。', 'muted');
    }
    state.cards = state.cards.filter((cs) => cs.status === 'shown');
  }

  function selectText(node) {
    const range = document.createRange();
    range.selectNodeContents(node);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (err) {
        /* 没有权限：退回旧办法 */
      }
    }
    const active = document.activeElement;
    const area = h('textarea', { className: 'offscreen', readonly: true, tabindex: '-1', 'aria-hidden': 'true' });
    area.value = text;
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (err) {
      ok = false;
    }
    area.remove();
    if (active && typeof active.focus === 'function') active.focus();
    return ok;
  }

  async function onSend(cs) {
    if (cs.busy || cs.status !== 'shown') return;
    setCardBusy(cs, true);
    try {
      if (!cs.manualCopy && !(await copyText(cs.card.text))) {
        cs.manualCopy = true;
        selectText(cs.text);
        cs.sendButton.textContent = '已手动复制，标记已发送';
        setStatus(
          cs,
          '浏览器不允许自动复制：推荐文字已选中，请按 Ctrl+C（手机上长按）复制，发给买家后再点「已手动复制，标记已发送」。',
          'warn',
        );
        return;
      }
      await api('POST', feedbackPath(cs, 'sent'), {});
      finishCard(cs, 'sent', '已复制并标记为已发送（计入采纳）。请粘贴到平台对话框发给买家。');
    } catch (err) {
      failCard(cs, err, '文字已复制，但没有记录成功：');
    } finally {
      setCardBusy(cs, false);
    }
  }

  function closePanel(cs) {
    cs.panel.replaceChildren();
    cs.actions.hidden = false;
    if (cs.lastButton && cs.status === 'shown') cs.lastButton.focus();
  }

  function openPanel(cs, form, opener, focusTarget) {
    cs.lastButton = opener;
    cs.panel.replaceChildren(form);
    cs.actions.hidden = true;
    form.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closePanel(cs);
      }
    });
    focusTarget.focus();
  }

  function openEdit(cs) {
    if (cs.busy || cs.status !== 'shown') return;
    const id = randomId('edit');
    const area = h('textarea', { id, rows: '4', maxlength: String(MAX_TEXT) });
    area.value = cs.card.text;
    const submit = h('button', { type: 'submit' }, '复制并提交修改');
    const cancel = h('button', { type: 'button', className: 'secondary', onclick: () => closePanel(cs) }, '取消');
    const form = h('form', { className: 'card-form', novalidate: true }, [
      h('label', { for: id }, '修改后的回答（会记录修改前后，生成候选知识等商家审核）'),
      area,
      h('div', { className: 'actions' }, [submit, cancel]),
    ]);
    let manualText = null;
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (cs.busy) return;
      const text = area.value.trim();
      if (!text) {
        setStatus(cs, '修改后的回答不能为空。', 'error');
        area.focus();
        return;
      }
      if (text.length > MAX_TEXT) {
        setStatus(cs, `修改后的回答最多 ${MAX_TEXT} 个字。`, 'error');
        area.focus();
        return;
      }
      setCardBusy(cs, true);
      try {
        if (manualText !== text && !(await copyText(text))) {
          manualText = text;
          area.focus();
          area.select();
          submit.textContent = '已手动复制，提交修改';
          setStatus(cs, '浏览器不允许自动复制：修改后的文字已选中，请手动复制，发给买家后再点「已手动复制，提交修改」。', 'warn');
          return;
        }
        const fb = await api('POST', feedbackPath(cs, 'edited'), { edited_text: text });
        let message = fb.status === 'sent' ? '内容没有改动，已按原样发送记录。' : '已复制并记录修改。';
        if (fb.candidate_knowledge_id) {
          message += `已生成候选知识 #${fb.candidate_knowledge_id}，等商家审核后可用于回答。`;
        } else if (fb.candidate_skipped) {
          message += `没有生成候选知识：${fb.candidate_skipped}。`;
        }
        cs.text.textContent = text; // 显示实际发出去的文字
        finishCard(cs, fb.status === 'sent' ? 'sent' : 'edited', message);
      } catch (err) {
        failCard(cs, err, '修改后的文字已复制，但没有记录成功：');
      } finally {
        setCardBusy(cs, false);
      }
    });
    openPanel(cs, form, cs.editButton, area);
  }

  function openReject(cs) {
    if (cs.busy || cs.status !== 'shown') return;
    const name = randomId('reason');
    const radios = Object.entries(LABELS.rejectReasons).map(([value, text], i) => {
      const id = `${name}-${i}`;
      return h('div', { className: 'radio' }, [
        h('input', { type: 'radio', name, id, value }),
        h('label', { for: id }, text),
      ]);
    });
    const intentId = randomId('intent');
    // 真实类型不能是推荐自己的类型（服务端会 422），不列出来
    const intentSelect = h('select', { id: intentId }, [
      h('option', { value: '' }, '不确定'),
      ...Object.entries(LABELS.intents)
        .filter(([value]) => value !== cs.card.intent)
        .map(([value, text]) => h('option', { value }, text)),
    ]);
    const intentField = h('div', { className: 'field', hidden: true }, [
      h('label', { for: intentId }, '买家真正问的是（选填，用于改进问题分类）'),
      intentSelect,
    ]);
    const fieldset = h('fieldset', null, [h('legend', null, '哪里不合适？'), ...radios]);
    const chosen = () => {
      const input = fieldset.querySelector('input[type="radio"]:checked');
      return input ? input.value : null;
    };
    fieldset.addEventListener('change', () => {
      intentField.hidden = chosen() !== 'wrong_intent';
    });
    const submit = h('button', { type: 'submit' }, '提交');
    const cancel = h('button', { type: 'button', className: 'secondary', onclick: () => closePanel(cs) }, '取消');
    const form = h('form', { className: 'card-form', novalidate: true }, [
      fieldset,
      intentField,
      h('div', { className: 'actions' }, [submit, cancel]),
    ]);
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (cs.busy) return;
      const reason = chosen();
      if (!reason) {
        setStatus(cs, '请选择不合适的原因。', 'error');
        const first = fieldset.querySelector('input');
        if (first) first.focus();
        return;
      }
      const body = { reason };
      if (reason === 'wrong_intent' && intentSelect.value) body.true_intent = intentSelect.value;
      setCardBusy(cs, true);
      try {
        const fb = await api('POST', feedbackPath(cs, 'rejected'), body);
        let message = `已记录：不合适（${label(LABELS.rejectReasons, reason)}）。`;
        if (fb.knowledge_gap) message += '已提醒商家补充知识。';
        if (fb.labeled) message += '已记入判断层标注集。';
        finishCard(cs, 'rejected', message);
      } catch (err) {
        failCard(cs, err, '没有记录成功：');
      } finally {
        setCardBusy(cs, false);
      }
    });
    openPanel(cs, form, cs.rejectButton, radios[0].querySelector('input'));
  }

  // ================================================================ 店铺数据（店长、商家：§12.4 报表、§5.2.3 缺失知识、附录 C 问卷）

  function sinceParam() {
    const days = Number($('range').value);
    if (!days) return '';
    return `?since=${enc(new Date(Date.now() - days * 86400000).toISOString())}`;
  }

  async function loadConsole() {
    if (!state.me || !CONSOLE_ROLES.includes(state.me.role)) return;
    const me = state.me;
    const run = nextRun('console');
    $('adoption').replaceChildren(h('p', { className: 'muted' }, '加载中…'));
    $('gaps').replaceChildren(h('p', { className: 'muted' }, '加载中…'));
    const [adoption, gaps] = await Promise.allSettled([
      api('GET', shopPath(`reports/adoption${sinceParam()}`)),
      api('GET', shopPath('knowledge/gaps')),
    ]);
    if (state.me !== me) return; // 期间登录失效或换人登录：不把上一位店铺的报表显示给下一位
    if (!latest('console', run)) return; // 期间又换了统计范围或提交了问卷：以后一次为准
    if (adoption.status === 'fulfilled') renderAdoption(adoption.value);
    else $('adoption').replaceChildren(h('p', { className: 'error-inline' }, adoption.reason.message));
    if (gaps.status === 'fulfilled') renderGaps(gaps.value);
    else $('gaps').replaceChildren(h('p', { className: 'error-inline' }, gaps.reason.message));
  }

  function statRow(name, value) {
    return [h('dt', null, name), h('dd', null, value)];
  }

  function reasonSummary(stats) {
    const reasons = stats.reject_reasons || {};
    const parts = Object.entries(reasons)
      .filter(([, count]) => count > 0)
      .map(([reason, count]) => `${label(LABELS.rejectReasons, reason)} ${count}`);
    return parts.length ? parts.join('、') : '—';
  }

  function renderAdoption(report) {
    const box = $('adoption');
    const o = report && report.overall;
    if (!o || !o.total) {
      box.replaceChildren(h('p', { className: 'muted' }, '这段时间还没有推荐记录。'));
      return;
    }
    const rate = o.group_adoption_rate;
    const reached = typeof rate === 'number' && rate >= ADOPTION_TARGET;
    const meter = h('meter', {
      min: '0',
      max: '1',
      low: String(ADOPTION_TARGET),
      high: '0.9',
      optimum: '1',
      value: String(typeof rate === 'number' ? rate : 0),
      'aria-label': '采纳率',
    });
    const headline = h('p', { className: 'headline' }, [
      h('strong', { className: reached ? 'good' : 'bad' }, pct(rate)),
      ` 采纳率（按买家消息算；Phase 0 目标 ≥ ${pct(ADOPTION_TARGET)}，§15）`,
    ]);
    const stats = h('dl', { className: 'stats' }, [
      ...statRow('推荐', `${o.total} 条 / ${o.groups} 组消息`),
      ...statRow('原样发送', `${o.sent}（${pct(o.as_is_rate)}）`),
      ...statRow('编辑后发送', `${o.edited_sent}（${pct(o.edited_rate)}）`),
      ...statRow('不合适', `${o.rejected}（${pct(o.reject_rate)}）`),
      ...statRow('已过期 / 未操作', `${o.expired} / ${o.shown}`),
      ...statRow('按条数的采纳率', pct(o.adoption_rate)),
      ...statRow('弃用原因', reasonSummary(o)),
    ]);
    const nodes = [headline, meter, stats];
    const byIntent = Object.values(report.by_intent || {});
    if (byIntent.length) {
      const head = h('tr', null, ['问题类型', '消息', '采纳率', '原样', '编辑后', '不合适', '弃用原因'].map((t) => h('th', { scope: 'col' }, t)));
      const rows = byIntent.map((s) =>
        h('tr', null, [
          h('th', { scope: 'row' }, s.label || label(LABELS.intents, s.intent)),
          h('td', null, String(s.groups)),
          h('td', null, pct(s.group_adoption_rate)),
          h('td', null, String(s.sent)),
          h('td', null, String(s.edited_sent)),
          h('td', null, String(s.rejected)),
          h('td', null, reasonSummary(s)),
        ]),
      );
      const table = h('table', null, [
        h('caption', null, '按问题类型'),
        h('thead', null, head),
        h('tbody', null, rows),
      ]);
      nodes.push(h('div', { className: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': '按问题类型的采纳率' }, table));
    }
    if (report.shadow_excluded) {
      nodes.push(h('p', { className: 'hint' }, `影子模式生成的 ${report.shadow_excluded} 条推荐不计入。`));
    }
    box.replaceChildren(...nodes);
  }

  function renderGaps(gaps) {
    const list = Array.isArray(gaps) ? gaps : [];
    if (!list.length) {
      $('gaps').replaceChildren(h('p', { className: 'muted' }, '暂时没有要补的知识。'));
      return;
    }
    $('gaps').replaceChildren(h('ul', { className: 'gaps' }, list.map((g) => h('li', null, g.description || g.marker))));
  }

  function buildQuestionnaire() {
    const fields = (CONFIG.questionnaire || []).map((q) => {
      const id = `q-${q.id}`;
      return h('div', { className: 'field' }, [
        h('label', { for: id }, `第 ${q.id} 题：${q.text}`),
        h('textarea', { id, rows: '2', maxlength: String(MAX_TEXT), 'data-qid': String(q.id) }),
      ]);
    });
    $('questionnaire-fields').replaceChildren(...fields);
  }

  async function submitQuestionnaire(event) {
    event.preventDefault();
    const status = $('questionnaire-status');
    const answers = {};
    for (const area of $('questionnaire-fields').querySelectorAll('textarea')) {
      const text = area.value.trim();
      if (text) answers[area.dataset.qid] = text;
    }
    if (!Object.keys(answers).length) {
      status.textContent = '请至少填写一题。';
      status.className = 'status error';
      return;
    }
    const button = $('questionnaire-button');
    button.disabled = true;
    const me = state.me;
    try {
      const result = await api('POST', shopPath('questionnaire'), { answers });
      if (state.me !== me) return;
      status.textContent = `已保存：写入政策 ${result.policies.length} 项，更新店铺配置 ${result.config.length} 项。`;
      status.className = 'status ok';
      loadConsole();
    } catch (err) {
      if (state.me !== me) return;
      status.textContent = `没有保存成功：${err.message}`;
      status.className = 'status error';
    } finally {
      if (state.me === me) button.disabled = false; // 期间退出过：退出时已经复位
    }
  }

  // ================================================================ 启动

  /**
   * 从往返缓存（bfcache）恢复：离开期间另一个页面退出或换人登录了（sessionStorage 里的令牌变了），
   * 本页内存里还是原来的身份。先丢掉它（不动 sessionStorage，那是另一个页面的），有令牌再用它登录。
   */
  function onPageShow(event) {
    if (!event.persisted) return;
    const stored = storage.get(STORAGE_TOKEN);
    if (stored === state.stored) return;
    const wasSignedIn = Boolean(state.stored);
    signOut({ keepStorage: true });
    if (stored) signIn(stored);
    else if (wasSignedIn) {
      showError('已在另一个页面退出登录，这个页面也已退出。请重新登录。');
      announce('已退出');
    }
  }

  function init() {
    if (CONFIG.platform_name) $('platform-name').textContent = CONFIG.platform_name;
    $('question').maxLength = MAX_TEXT;
    buildQuestionnaire();

    $('login-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const token = $('token').value.trim();
      if (!token) {
        showError('请输入访问令牌。');
        $('token').focus();
        return;
      }
      signIn(token);
    });
    $('sign-out').addEventListener('click', () => {
      signOut();
      clearError();
      announce('已退出');
    });
    window.addEventListener('pageshow', onPageShow);
    $('new-session').addEventListener('click', () => {
      newSession();
      $('question').focus();
      announce('已开始新会话');
    });
    $('ask-form').addEventListener('submit', (event) => {
      event.preventDefault();
      ask();
    });
    $('question').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        ask();
      }
    });
    $('item-filter').addEventListener('input', renderItemOptions);
    $('report-form').addEventListener('submit', (event) => {
      event.preventDefault();
      loadConsole();
    });
    $('questionnaire-form').addEventListener('submit', submitQuestionnaire);

    const saved = storage.get(STORAGE_TOKEN);
    if (saved) signIn(saved);
    else $('token').focus();
  }

  init();
})();
