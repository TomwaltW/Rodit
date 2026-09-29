/*
 * Rodit 商家控制台（方案 §9.1 知识编辑、配置、报表；§5.2 知识中心；§9.3 配置分层；§13.4 上线节奏；
 * §13.5 每周复核；§12.4 报表）。
 *
 * 约定（和话术助手页 assist.js 一样）：
 * - 服务端和用户给的文字一律用 textContent / 文本节点写进页面，从不按 HTML 解析（防 XSS）；
 * - 令牌只放 sessionStorage，只放进 Authorization 请求头，不进网址、不打日志；
 *   和话术助手页用同一个键，同一标签页里两个页面互相跳转不用重新登录；
 * - 接口地址用相对路径；店铺接口一律写成 shopPath('字面路径', 查询参数)，测试据此核对接口清单；
 * - 每次登录换一个 state.me 对象：响应回来时已经退出或换人登录就丢掉，不写进下一位的界面；
 *   401 只在发请求的令牌还是当前令牌时才登出；
 * - 错误码和说明来自接口的统一错误格式 {"error", "message"}（rodit.api.errors）；
 * - 页签按角色显示（配置块 tabs），写操作的按钮只给商家（配置块 role_groups.merchant），
 *   真正的权限检查在接口上（§9.2）。
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

  const L = {
    roles: CONFIG.roles || {},
    intents: CONFIG.intents || {},
    knowledgeStatus: CONFIG.knowledge_status || {},
    knowledgeSource: CONFIG.knowledge_source || {},
    itemFields: CONFIG.item_fields || {},
    syncFields: CONFIG.sync_fields || {},
    stages: CONFIG.stages || {},
    gateStatus: CONFIG.gate_status || {},
    gateChecks: CONFIG.gate_checks || {},
    dimensions: CONFIG.dimensions || {},
    scoreGuide: CONFIG.score_guide || {},
    reviewStatus: CONFIG.review_status || {},
    suggestionStatus: CONFIG.suggestion_status || {},
    kinds: CONFIG.kinds || {},
    rejectReasons: CONFIG.reject_reasons || {},
    hardRules: CONFIG.hard_rules || {},
    degraded: CONFIG.degraded || {},
  };
  const GROUPS = CONFIG.role_groups || {};
  const TAB_ROLES = CONFIG.tabs || {};
  const DEFAULT_TAB = CONFIG.default_tab || {};
  const LIMITS = CONFIG.limits || {};
  const ITEM_LIMITS = CONFIG.item_limits || {};
  const CONFIG_FIELDS = Array.isArray(CONFIG.config_fields) ? CONFIG.config_fields : [];
  const FULL_SCORE = typeof CONFIG.full_score === 'number' ? CONFIG.full_score : 2;
  const ADOPTION_TARGET = typeof CONFIG.adoption_target === 'number' ? CONFIG.adoption_target : 0.5;
  const PERIOD_RE = (() => {
    try {
      return new RegExp(CONFIG.period_pattern || '^\\d{4}-W\\d{2}$');
    } catch (err) {
      return /^\d{4}-W\d{2}$/;
    }
  })();

  const STORAGE_TOKEN = 'rodit.assist.token'; // 和话术助手页同一个键（同一标签页里切换页面不用重新登录）
  const STORAGE_SESSION = 'rodit.assist.session'; // 话术助手页的会话编号：退出时一并清掉
  const STORAGE_TAB = 'rodit.console.tab'; // 界面偏好：上次看的页签

  const state = {
    token: null,
    me: null,
    signingIn: false,
    tab: null,
    loaded: {},
    items: [], // 商品卡片列表（知识页的商品下拉框、商品页共用）
    policies: [],
    config: null, // 最近一次的店铺配置视图
    stageMessage: null, // 切换阶段成功后，刷新上线状态时显示的结果
    review: { period: null, items: [] },
    itemOpener: null,
  };

  // ================================================================ 小工具

  /** 建元素。props.text 用 textContent；children（可嵌套数组）里的字符串是文本节点。不接受 HTML 字符串。 */
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
    const list = children === undefined ? [] : [].concat(children).flat(Infinity);
    for (const child of list) {
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

  const enc = encodeURIComponent;
  const has = (map, key) => Object.prototype.hasOwnProperty.call(map, key);
  const inGroup = (group, role) => Array.isArray(GROUPS[group]) && GROUPS[group].includes(role);
  const isMerchant = () => Boolean(state.me) && inGroup('merchant', state.me.role);
  const hasText = (value) => typeof value === 'string' && value.trim() !== '';

  let seq = 0;
  const uid = (prefix) => `${prefix}-${(seq += 1)}`;

  function label(map, key) {
    return has(map, key) ? map[key] : String(key);
  }

  function pct(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    return `${Math.round(value * 1000) / 10}%`;
  }

  function num(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 100) / 100);
  }

  const TIME_FORMAT = (() => {
    try {
      return new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hourCycle: 'h23',
      });
    } catch (err) {
      return null;
    }
  })();

  /** 时间 → 东八区的 "2026-09-28 10:35"（withTime=false 时只要日期）。 */
  function fmtTime(value, withTime = true) {
    if (!value) return '—';
    const date = value instanceof Date ? value : new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    if (!TIME_FORMAT) return date.toISOString().replace('T', ' ').slice(0, withTime ? 16 : 10);
    const p = {};
    for (const part of TIME_FORMAT.formatToParts(date)) p[part.type] = part.value;
    const day = `${p.year}-${p.month}-${p.day}`;
    return withTime ? `${day} ${p.hour}:${p.minute}` : day;
  }

  const fmtDate = (value) => fmtTime(value, false);

  /** [since, until) 的日期范围（until 不含，显示成前一天）。 */
  function fmtRange(since, until) {
    if (!since || !until) return '—';
    const end = new Date(new Date(until).getTime() - 1);
    return `${fmtDate(since)} – ${fmtDate(end)}`;
  }

  /** 一行一条的文字 → 列表：去掉首尾空白和空行，去重保序。 */
  function linesOf(text) {
    const out = [];
    for (const raw of String(text || '').split(/\r?\n/)) {
      const line = raw.trim();
      if (line && !out.includes(line)) out.push(line);
    }
    return out;
  }

  const sameList = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

  let liveTimer = null;
  function announce(message) {
    const live = $('live');
    live.textContent = '';
    window.clearTimeout(liveTimer);
    liveTimer = window.setTimeout(() => {
      live.textContent = message;
    }, 50);
  }

  /** 页面顶部的错误区（role=alert）。link 可选：{href, text}，比如去另一个页面。 */
  function showError(message, link) {
    const box = $('error');
    box.replaceChildren(message);
    if (link) box.append(' ', h('a', { href: link.href }, link.text));
    box.hidden = false;
  }

  function clearError() {
    const box = $('error');
    box.replaceChildren();
    box.hidden = true;
  }

  function setStatus(node, message, kind) {
    if (!node) return;
    node.textContent = message;
    node.className = `status ${kind || ''}`.trim();
  }

  /** 操作失败：就近的状态行 + 页面顶部的错误区（读屏会念出来）。 */
  function reportFailure(statusNode, prefix, err) {
    const message = `${prefix}${err.message}`;
    setStatus(statusNode, message, 'error');
    showError(message);
  }

  /** 防重复提交：同一个 ctx 上一次还没完成时直接返回；执行期间禁用 controls。 */
  async function busy(ctx, controls, task) {
    if (ctx.busy) return undefined;
    ctx.busy = true;
    const list = controls.filter(Boolean);
    for (const control of list) control.disabled = true;
    try {
      return await task();
    } finally {
      ctx.busy = false;
      for (const control of list) control.disabled = false;
    }
  }

  function loading(node) {
    node.setAttribute('aria-busy', 'true');
    node.replaceChildren(h('p', { className: 'muted' }, '加载中…'));
  }

  function put(node, ...children) {
    node.removeAttribute('aria-busy');
    node.replaceChildren(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  }

  function loadFailed(node, what, err) {
    const message = `${what}没有加载出来：${err.message}`;
    put(node, h('p', { className: 'error-inline' }, message));
    showError(message);
  }

  function statRow(name, value) {
    return [h('dt', null, name), h('dd', null, value)];
  }

  /** 表格（窄屏时在自己的区域里横向滚动）。hideCaption：上面已经有同名标题时，标题只给读屏用。 */
  function table(caption, headers, rows, hideCaption) {
    return h('div', { className: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': caption }, [
      h('table', null, [
        h('caption', { className: hideCaption ? 'visually-hidden' : null }, caption),
        h('thead', null, h('tr', null, headers.map((t) => h('th', { scope: 'col' }, t)))),
        h('tbody', null, rows),
      ]),
    ]);
  }

  function field(id, text, control, hint) {
    const hintId = hint ? `${id}-hint` : null;
    if (hintId) control.setAttribute('aria-describedby', hintId);
    return h('div', { className: 'field' }, [
      h('label', { for: id }, text),
      control,
      hint ? h('p', { id: hintId, className: 'hint' }, hint) : null,
    ]);
  }

  function fillOptions(select, labels) {
    for (const [value, text] of Object.entries(labels)) select.append(h('option', { value }, text));
  }

  // ================================================================ 接口

  class ApiFailure extends Error {
    constructor(status, code, message, data) {
      super(message);
      this.status = status;
      this.code = code;
      this.data = data || null; // 完整的错误响应（409 gate_not_passed 带门槛结果）
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
      const failure = new ApiFailure(resp.status, code, describeFailure(resp.status, data && data.message), data);
      if (resp.status === 401 && state.me && state.token === token) {
        signOut();
        showError(`${failure.message}。请重新登录。`);
      }
      throw failure;
    }
    return data;
  }

  /** 本店铺的接口地址。query 里空的参数不带。 */
  function shopPath(suffix, query) {
    const me = state.me;
    if (!me) throw new ApiFailure(0, 'signed_out', '已退出登录');
    let path = `v1/shops/${enc(me.shop_id)}/${suffix}`;
    if (query) {
      const params = new URLSearchParams();
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined && value !== null && value !== '') params.set(key, String(value));
      }
      const text = params.toString();
      if (text) path += `?${text}`;
    }
    return path;
  }

  // ================================================================ 登录、页签

  const TAB_IDS = Array.from(document.querySelectorAll('#tabs [role="tab"]'), (b) => b.dataset.tab);

  const tabButtons = () => Array.from(document.querySelectorAll('#tabs [role="tab"]'));
  const tabsFor = (role) => TAB_IDS.filter((id) => Array.isArray(TAB_ROLES[id]) && TAB_ROLES[id].includes(role));

  async function signIn(token) {
    if (state.signingIn) return false;
    state.signingIn = true;
    clearError();
    state.token = token;
    const button = $('login-button');
    button.disabled = true;
    let me;
    try {
      me = await api('GET', 'v1/me');
    } catch (err) {
      state.token = null;
      // 令牌无效才丢掉；网络、服务故障时留着，刷新后还能直接登录
      if (err.status === 401 || err.status === 403) storage.remove(STORAGE_TOKEN);
      if (err.code === 'forbidden_role') {
        showError('这个令牌是平台对接用的，不能登录网页工具；请用店长、商家或复核人员的令牌。');
      } else {
        showError(err.message);
      }
      $('token').focus();
      return false;
    } finally {
      state.signingIn = false;
      button.disabled = false;
    }
    if (!me || !inGroup('person', me.role)) {
      state.token = null;
      storage.remove(STORAGE_TOKEN);
      showError('这个令牌不能登录网页工具。');
      $('token').focus();
      return false;
    }
    if (!tabsFor(me.role).length) {
      // 客服：控制台没有可用的功能。令牌留着，同一标签页打开话术助手时直接登录
      state.token = null;
      storage.set(STORAGE_TOKEN, token);
      $('token').value = '';
      showError(`${label(L.roles, me.role)}请使用话术助手，控制台只给店长、商家和复核人员用。`, {
        href: 'assist',
        text: '打开话术助手',
      });
      return false;
    }
    state.me = me;
    storage.set(STORAGE_TOKEN, token);
    enterApp();
    return true;
  }

  function enterApp() {
    const me = state.me;
    $('token').value = '';
    $('login').hidden = true;
    $('app').hidden = false;
    $('whoami').hidden = false;
    $('identity').textContent = `${label(L.roles, me.role)} ${me.user_id} · 店铺 ${me.shop_id}`;
    const merchant = isMerchant();
    for (const node of document.querySelectorAll('[data-write]')) node.hidden = !merchant;
    for (const node of document.querySelectorAll('[data-readonly]')) node.hidden = merchant;
    const tabs = tabsFor(me.role);
    for (const button of tabButtons()) button.hidden = !tabs.includes(button.dataset.tab);
    state.loaded = {};
    const saved = storage.get(STORAGE_TAB);
    const preferred = DEFAULT_TAB[me.role];
    const first = tabs.includes(saved) ? saved : tabs.includes(preferred) ? preferred : tabs[0];
    selectTab(first);
    $(`tab-${first}`).focus();
    announce(`已登录：${$('identity').textContent}`);
  }

  function signOut() {
    state.token = null;
    state.me = null;
    state.tab = null;
    state.loaded = {};
    state.items = [];
    state.policies = [];
    state.config = null;
    state.stageMessage = null;
    state.review = { period: null, items: [] };
    state.itemOpener = null;
    storage.remove(STORAGE_TOKEN);
    storage.remove(STORAGE_SESSION);
    storage.remove(STORAGE_TAB);
    for (const node of document.querySelectorAll('[data-dyn]')) put(node);
    for (const node of document.querySelectorAll('#app .status')) setStatus(node, '', '');
    $('item-detail').hidden = true;
    delete $('item-detail').dataset.itemId;
    for (const form of $('app').querySelectorAll('form')) form.reset();
    $('faq-item').replaceChildren(h('option', { value: '' }, '通用（不限商品）'));
    $('faq-create').open = false;
    for (const button of tabButtons()) {
      button.hidden = true;
      button.setAttribute('aria-selected', 'false');
      button.tabIndex = -1;
    }
    for (const panel of document.querySelectorAll('[role="tabpanel"]')) panel.hidden = true;
    $('app').hidden = true;
    $('whoami').hidden = true;
    $('identity').textContent = '';
    $('login').hidden = false;
    $('token').focus();
  }

  function selectTab(id) {
    if (!state.me || !tabsFor(state.me.role).includes(id)) return;
    for (const button of tabButtons()) {
      const on = button.dataset.tab === id;
      button.setAttribute('aria-selected', on ? 'true' : 'false');
      button.tabIndex = on ? 0 : -1;
      $(button.getAttribute('aria-controls')).hidden = !on;
    }
    state.tab = id;
    storage.set(STORAGE_TAB, id);
    if (!state.loaded[id]) {
      state.loaded[id] = true;
      LOADERS[id]();
    }
  }

  /** 写操作影响了别的页签的数据（比如补了政策，「要补的知识」会变）：下次切过去时重新加载。 */
  function stale(...ids) {
    for (const id of ids) if (id !== state.tab) state.loaded[id] = false;
  }

  /** 页签的键盘操作（WAI-ARIA tabs）：左右方向键、Home、End 切换并聚焦。 */
  function onTabKey(event) {
    const tabs = tabButtons().filter((b) => !b.hidden);
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    let next = null;
    if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
    else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length];
    else if (event.key === 'Home') next = tabs[0];
    else if (event.key === 'End') next = tabs[tabs.length - 1];
    if (!next) return;
    event.preventDefault();
    selectTab(next.dataset.tab);
    next.focus();
  }

  // ================================================================ 商品列表（知识页的下拉框、商品页共用）

  async function fetchItems() {
    const me = state.me;
    const list = await api('GET', shopPath('console/items'));
    if (state.me === me) {
      state.items = Array.isArray(list) ? list : [];
      renderItemOptions();
    }
    return list;
  }

  function itemTitle(itemId) {
    const item = state.items.find((i) => i.item_id === itemId);
    return item ? `${item.title}（${item.item_id}）` : itemId;
  }

  function itemOptions(current) {
    const options = [h('option', { value: '' }, '通用（不限商品）')];
    for (const item of state.items) options.push(h('option', { value: item.item_id }, itemTitle(item.item_id)));
    if (current && !state.items.some((i) => i.item_id === current)) {
      options.push(h('option', { value: current }, current)); // 卡片已经不在了：保留原值
    }
    return options;
  }

  function renderItemOptions() {
    const select = $('faq-item');
    const current = select.value;
    select.replaceChildren(...itemOptions(current));
    select.value = current;
  }

  // ================================================================ 知识（§5.2、§5.8）

  const faqCreateCtx = { busy: false };

  async function loadKnowledge() {
    const me = state.me;
    for (const id of ['gaps', 'candidates', 'faqs']) loading($(id));
    const [gaps, candidates, faqs] = await Promise.allSettled([
      api('GET', shopPath('knowledge/gaps')),
      api('GET', shopPath('console/faqs', { status: 'candidate' })),
      api('GET', shopPath('console/faqs', { status: $('faq-status').value })),
      fetchItems().catch(() => null), // 下拉框和商品名用；失败时显示商品 ID
    ]);
    if (state.me !== me) return; // 期间退出或换人登录：不把上一位店铺的数据显示给下一位
    if (gaps.status === 'fulfilled') renderGaps(gaps.value);
    else loadFailed($('gaps'), '要补的知识', gaps.reason);
    if (candidates.status === 'fulfilled') renderFaqList($('candidates'), candidates.value, '没有待审核的候选知识。');
    else loadFailed($('candidates'), '候选知识', candidates.reason);
    if (faqs.status === 'fulfilled') renderFaqList($('faqs'), faqs.value, '没有符合条件的 FAQ。');
    else loadFailed($('faqs'), 'FAQ 列表', faqs.reason);
  }

  function renderGaps(gaps) {
    const list = Array.isArray(gaps) ? gaps : [];
    if (!list.length) {
      put($('gaps'), h('p', { className: 'muted' }, '暂时没有要补的知识。'));
      return;
    }
    put($('gaps'), h('ul', { className: 'gaps' }, list.map((g) => h('li', null, g.description || g.marker))));
  }

  function renderFaqList(box, list, emptyText) {
    const rows = Array.isArray(list) ? list : [];
    if (!rows.length) {
      put(box, h('p', { className: 'muted' }, emptyText));
      return;
    }
    put(box, h('ul', { className: 'faq-list' }, rows.map((faq) => h('li', null, faqEntry(faq)))));
  }

  function faqEntry(faq) {
    const titleId = uid('faq');
    const entry = {
      faq,
      busy: false,
      actions: h('div', { className: 'actions tight' }),
      panel: h('div', { className: 'entry-panel' }),
      status: h('p', { className: 'status', role: 'status', tabindex: '-1' }),
      buttons: [],
      opener: null,
    };
    const byline = [`创建于 ${fmtTime(faq.created_at)}`];
    if (faq.approved_by) byline.push(`审核人 ${faq.approved_by}`);
    const article = h('article', { className: `faq st-${faq.status}`, 'aria-labelledby': titleId }, [
      h('h4', { id: titleId, className: 'faq-title' }, [h('span', { className: 'faq-id' }, `#${faq.id}`), ' ', faq.question]),
      h('p', { className: 'faq-meta' }, [
        h('span', { className: `badge st-${faq.status}` }, label(L.knowledgeStatus, faq.status)),
        h('span', { className: 'badge' }, label(L.knowledgeSource, faq.source)),
        h('span', { className: 'badge' }, faq.intent ? label(L.intents, faq.intent) : '不限问题类型'),
        h('span', { className: 'badge' }, faq.item_id ? `商品：${itemTitle(faq.item_id)}` : '通用'),
      ]),
      h('p', { className: 'faq-answer' }, faq.answer),
      h('p', { className: 'hint' }, byline.join(' · ')),
      entry.actions,
      entry.panel,
      entry.status,
    ]);
    const button = (text, onClick, secondary) => {
      const b = h('button', { type: 'button', className: secondary ? 'secondary' : null, 'aria-describedby': titleId }, text);
      b.addEventListener('click', () => onClick(b));
      entry.buttons.push(b);
      return b;
    };
    if (isMerchant() && faq.status === 'approved') {
      entry.actions.append(
        button('修改', (b) => openFaqEditor(entry, 'edit', b), true),
        button('下线', (b) => openRetire(entry, b), true),
      );
    } else if (isMerchant() && faq.status === 'candidate') {
      entry.actions.append(
        button('审核通过', (b) => approveCandidate(entry, b)),
        button('修改后通过', (b) => openFaqEditor(entry, 'approve', b), true),
        button('驳回', (b) => openRetire(entry, b), true),
      );
    }
    return article;
  }

  function openPanel(entry, content, opener, focusTarget) {
    if (entry.busy) return;
    entry.opener = opener;
    entry.panel.replaceChildren(content);
    entry.actions.hidden = true;
    content.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closePanel(entry);
      }
    });
    focusTarget.focus();
  }

  function closePanel(entry) {
    entry.panel.replaceChildren();
    entry.actions.hidden = false;
    if (entry.opener && entry.opener.isConnected) entry.opener.focus();
  }

  function intentSelect(id, current) {
    const select = h('select', { id }, [h('option', { value: '' }, '不限')]);
    fillOptions(select, L.intents);
    select.value = current || '';
    return select;
  }

  function itemSelect(id, current) {
    const select = h('select', { id }, itemOptions(current));
    select.value = current || '';
    return select;
  }

  /** 表单里的 FAQ 内容 → 校验结果；baseline 给出时只留有改动的字段（null 表示清空商品 / 问题类型）。 */
  function faqBody(values, baseline) {
    const question = values.question.trim();
    const answer = values.answer.trim();
    if (!question) return { error: '买家问题不能为空。', focus: 'question' };
    if (!answer) return { error: '标准回答不能为空。', focus: 'answer' };
    if (question.length > LIMITS.question) return { error: `买家问题最多 ${LIMITS.question} 个字。`, focus: 'question' };
    if (answer.length > LIMITS.answer) return { error: `标准回答最多 ${LIMITS.answer} 个字。`, focus: 'answer' };
    const next = { question, answer, item_id: values.item_id || null, intent: values.intent || null };
    const body = {};
    for (const [key, value] of Object.entries(next)) {
      if (!baseline) {
        if (value !== null) body[key] = value;
      } else if (value !== (baseline[key] || null)) {
        body[key] = value;
      }
    }
    return { body };
  }

  function warningsBox(warnings) {
    const list = Array.isArray(warnings) ? warnings : [];
    if (!list.length) return null;
    const blocking = list.filter((w) => w.blocks_answer).length;
    const title = blocking
      ? `已保存，但回答时用不上：有 ${blocking} 条会被质检拦截或被店铺政策覆盖，建议修改。`
      : `已保存，有 ${list.length} 条提醒：`;
    return h('div', { className: blocking ? 'warnings blocking' : 'warnings', role: 'group', 'aria-label': '保存时的提醒' }, [
      h('p', { className: 'warnings-title' }, title),
      h(
        'ul',
        null,
        list.map((w) =>
          h('li', { className: w.blocks_answer ? 'blocks' : 'warn' }, [
            h('span', { className: 'badge' }, w.blocks_answer ? '回答时用不上' : '提醒'),
            ' ',
            h('span', { className: 'muted' }, `${label(L.hardRules, w.rule)}：`),
            w.detail,
          ]),
        ),
      ),
    ]);
  }

  function showFaqChange(target, change, action) {
    const k = change.knowledge;
    const old = change.retired;
    let text;
    if (action === 'create') text = `已保存并上线：FAQ #${k.id}。`;
    else if (action === 'edit' && old) {
      text = `已修改：旧条目 #${old.id} 已下线，改过的内容作为新条目 #${k.id} 上线（已有推荐的依据仍指向 #${old.id}）。`;
    } else if (old) text = `已按修改后的内容上线：新条目 #${k.id}，候选 #${old.id} 已下线。`;
    else text = `已审核通过：#${k.id} 已上线。`;
    const warnings = Array.isArray(change.warnings) ? change.warnings : [];
    put(target, h('p', { className: 'status ok' }, text), warningsBox(warnings));
    target.focus();
    announce(warnings.length ? `${text}有 ${warnings.length} 条提醒。` : text);
  }

  async function createFaq(event) {
    event.preventDefault();
    if (!isMerchant()) return;
    const result = $('faq-result');
    const parsed = faqBody({
      question: $('faq-question').value,
      answer: $('faq-answer').value,
      item_id: $('faq-item').value,
      intent: $('faq-intent').value,
    });
    if (parsed.error) {
      put(result, h('p', { className: 'status error' }, parsed.error));
      $(parsed.focus === 'question' ? 'faq-question' : 'faq-answer').focus();
      return;
    }
    const me = state.me;
    clearError();
    await busy(faqCreateCtx, [$('faq-submit')], async () => {
      try {
        const change = await api('POST', shopPath('console/faqs'), parsed.body);
        if (state.me !== me) return;
        $('faq-form').reset();
        showFaqChange(result, change, 'create');
        await loadKnowledge();
      } catch (err) {
        if (state.me !== me) return;
        put(result, h('p', { className: 'status error' }, `没有保存成功：${err.message}`));
        showError(`FAQ 没有保存成功：${err.message}`);
      }
    });
  }

  function openFaqEditor(entry, mode, opener) {
    const faq = entry.faq;
    const ids = { q: uid('fq'), a: uid('fa'), item: uid('fi'), intent: uid('ft') };
    const q = h('textarea', { id: ids.q, rows: '2', maxlength: String(LIMITS.question) });
    q.value = faq.question;
    const a = h('textarea', { id: ids.a, rows: '3', maxlength: String(LIMITS.answer) });
    a.value = faq.answer;
    const item = itemSelect(ids.item, faq.item_id);
    const intent = intentSelect(ids.intent, faq.intent);
    const submit = h('button', { type: 'submit' }, mode === 'edit' ? '保存修改' : '按修改后的内容通过');
    const cancel = h('button', { type: 'button', className: 'secondary', onclick: () => closePanel(entry) }, '取消');
    const form = h('form', { className: 'entry-form', novalidate: true }, [
      h(
        'p',
        { className: 'hint' },
        mode === 'edit'
          ? '保存后旧条目下线，改过的内容作为新条目上线（已有推荐的依据仍指向旧条目）。'
          : '有改动时候选条目下线、改过的内容作为新条目上线；不用改就直接点「审核通过」。',
      ),
      field(ids.q, '买家问题', q),
      field(ids.a, '标准回答', a),
      h('div', { className: 'field-row' }, [field(ids.item, '适用商品', item), field(ids.intent, '问题类型', intent)]),
      h('div', { className: 'actions tight' }, [submit, cancel]),
    ]);
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const parsed = faqBody({ question: q.value, answer: a.value, item_id: item.value, intent: intent.value }, faq);
      if (parsed.error) {
        setStatus(entry.status, parsed.error, 'error');
        (parsed.focus === 'question' ? q : a).focus();
        return;
      }
      if (!Object.keys(parsed.body).length) {
        setStatus(entry.status, mode === 'edit' ? '没有改动。' : '没有改动；不用改就直接点「审核通过」。', 'warn');
        return;
      }
      const me = state.me;
      clearError();
      await busy(entry, [submit, cancel], async () => {
        try {
          const id = enc(String(faq.id));
          const change =
            mode === 'edit'
              ? await api('POST', shopPath(`console/faqs/${id}/edit`), parsed.body)
              : await api('POST', shopPath(`console/candidates/${id}/approve-edited`), parsed.body);
          if (state.me !== me) return;
          showFaqChange($('knowledge-result'), change, mode);
          await loadKnowledge();
        } catch (err) {
          if (state.me !== me) return;
          entryFailed(entry, err);
        }
      });
    });
    openPanel(entry, form, opener, q);
  }

  /** 列表上的操作失败：404 / 409 说明条目状态已经变了（别处处理过），刷新列表。 */
  function entryFailed(entry, err) {
    reportFailure(entry.status, '没有成功：', err);
    if (err.status === 404 || err.status === 409) loadKnowledge();
  }

  async function runEntryAction(entry, controls, task) {
    const me = state.me;
    clearError();
    await busy(entry, controls.concat(entry.buttons), async () => {
      try {
        const message = await task();
        if (state.me !== me) return;
        put($('knowledge-result'), h('p', { className: 'status ok' }, message));
        $('knowledge-result').focus();
        announce(message);
        await loadKnowledge();
      } catch (err) {
        if (state.me !== me) return;
        entryFailed(entry, err);
      }
    });
  }

  function approveCandidate(entry, button) {
    runEntryAction(entry, [button], async () => {
      const item = await api('POST', shopPath(`knowledge/candidates/${enc(String(entry.faq.id))}/approve`), {});
      return `已审核通过：#${item.id} 已上线。`;
    });
  }

  function openRetire(entry, opener) {
    const candidate = entry.faq.status === 'candidate';
    const confirm = h('button', { type: 'button', className: 'danger' }, candidate ? '确认驳回' : '确认下线');
    const cancel = h('button', { type: 'button', className: 'secondary', onclick: () => closePanel(entry) }, '取消');
    const box = h('div', { className: 'confirm' }, [
      h(
        'p',
        null,
        candidate ? '驳回后这条候选知识不会用于回答。' : '下线后这条 FAQ 不再用于回答（已有推荐的依据不受影响）。',
      ),
      h('div', { className: 'actions tight' }, [confirm, cancel]),
    ]);
    confirm.addEventListener('click', () =>
      runEntryAction(entry, [confirm, cancel], async () => {
        const item = await api('POST', shopPath(`knowledge/${enc(String(entry.faq.id))}/retire`), {});
        return candidate ? `已驳回候选知识 #${item.id}。` : `已下线 FAQ #${item.id}。`;
      }),
    );
    openPanel(entry, box, opener, confirm);
  }

  // ================================================================ 店铺政策（§5.2.1 优先级 2）

  async function loadPolicies() {
    const me = state.me;
    loading($('policies'));
    put($('policy-summary'));
    try {
      const rows = await api('GET', shopPath('console/policies'));
      if (state.me !== me) return;
      state.policies = Array.isArray(rows) ? rows : [];
      renderPolicySummary();
      put($('policies'), h('ul', { className: 'policy-list' }, state.policies.map((row) => policyRow(row))));
    } catch (err) {
      if (state.me !== me) return;
      loadFailed($('policies'), '店铺政策', err);
    }
  }

  function renderPolicySummary() {
    const known = state.policies.filter((r) => r.known);
    const filled = known.filter((r) => hasText(r.value)).length;
    put(
      $('policy-summary'),
      `已填写 ${filled} / ${known.length} 项。`,
      filled < known.length ? '没填的政策，买家问到时推荐会缺依据（「知识」页的「要补的知识」会列出来）。' : '',
    );
  }

  function policyRow(row, message) {
    const titleId = uid('policy');
    const filled = hasText(row.value);
    const meta = [row.key];
    if (row.questionnaire_id) meta.push(`问卷第 ${row.questionnaire_id} 题`);
    if (row.updated_at) meta.push(`更新于 ${fmtTime(row.updated_at)}`);
    const ctx = {
      busy: false,
      row,
      actions: h('div', { className: 'actions tight' }),
      panel: h('div', { className: 'entry-panel' }),
      status: h('p', { className: 'status', role: 'status', tabindex: '-1' }),
      opener: null,
      node: null,
    };
    ctx.node = h('li', { className: filled ? 'policy' : 'policy unfilled', 'aria-labelledby': titleId }, [
      h('div', { className: 'policy-head' }, [
        h('h3', { id: titleId, className: 'policy-label' }, row.label || row.key),
        h('span', { className: 'hint' }, meta.join(' · ')),
      ]),
      filled ? h('p', { className: 'policy-value' }, row.value) : h('p', { className: 'policy-value empty' }, '未填写'),
      row.known ? null : h('p', { className: 'hint' }, '旧数据：不是现有的政策项，只能查看。'),
      ctx.actions,
      ctx.panel,
      ctx.status,
    ]);
    if (isMerchant() && row.known) {
      const edit = h('button', { type: 'button', className: 'secondary', 'aria-describedby': titleId }, filled ? '修改' : '填写');
      edit.addEventListener('click', () => openPolicyEditor(ctx, edit));
      ctx.actions.append(edit);
    }
    if (message) setStatus(ctx.status, message, 'ok');
    return ctx.node;
  }

  function openPolicyEditor(ctx, opener) {
    const id = uid('policy-edit');
    const area = h('textarea', { id, rows: '3', maxlength: String(LIMITS.policy) });
    area.value = ctx.row.value || '';
    const save = h('button', { type: 'submit' }, '保存');
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const form = h('form', { className: 'entry-form', novalidate: true }, [
      field(id, `${ctx.row.label || ctx.row.key}（保存后即时生效）`, area),
      h('div', { className: 'actions tight' }, [save, cancel]),
    ]);
    const close = () => {
      ctx.panel.replaceChildren();
      ctx.actions.hidden = false;
      opener.focus();
    };
    cancel.addEventListener('click', close);
    form.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
      }
    });
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const value = area.value.trim();
      if (!value) {
        setStatus(ctx.status, '政策内容不能为空。', 'error');
        area.focus();
        return;
      }
      if (value.length > LIMITS.policy) {
        setStatus(ctx.status, `政策内容最多 ${LIMITS.policy} 个字。`, 'error');
        area.focus();
        return;
      }
      if (value === (ctx.row.value || '')) {
        setStatus(ctx.status, '没有改动。', 'warn');
        return;
      }
      const me = state.me;
      clearError();
      await busy(ctx, [save, cancel], async () => {
        try {
          const row = await api('POST', shopPath(`console/policies/${enc(ctx.row.key)}`), { value });
          if (state.me !== me) return;
          state.policies = state.policies.map((r) => (r.key === row.key ? row : r));
          renderPolicySummary();
          stale('knowledge');
          const message = `已保存「${row.label}」，即时生效。`;
          const fresh = policyRow(row, message);
          ctx.node.replaceWith(fresh);
          fresh.querySelector('.status').focus();
          announce(message);
        } catch (err) {
          if (state.me !== me) return;
          reportFailure(ctx.status, '没有保存成功：', err);
        }
      });
    });
    ctx.actions.hidden = true;
    ctx.panel.replaceChildren(form);
    area.focus();
  }

  // ================================================================ 商品卡片与同步（§5.2.2、§5.2.3）

  const syncCtx = { busy: false };

  async function loadItemsTab() {
    const me = state.me;
    loading($('items'));
    try {
      await fetchItems();
      if (state.me !== me) return;
      renderItemTable();
    } catch (err) {
      if (state.me !== me) return;
      loadFailed($('items'), '商品列表', err);
    }
  }

  function renderItemTable() {
    const list = state.items;
    if (!list.length) {
      put($('items'), h('p', { className: 'muted' }, '还没有商品卡片。商家可以点「同步商品」从平台拉取。'));
      return;
    }
    const fields = Object.keys(L.itemFields);
    const rows = list.map((item) => {
      const rowId = uid('item-row');
      const filled = new Set(Array.isArray(item.filled) ? item.filled : []);
      const open = h('button', { type: 'button', className: 'secondary', 'aria-describedby': rowId }, isMerchant() ? '查看 / 编辑' : '查看');
      open.addEventListener('click', () => loadItemDetail(item.item_id, open));
      return h('tr', { 'data-item': item.item_id }, [
        h('th', { scope: 'row', id: rowId, className: 'wrap' }, [item.title, h('span', { className: 'hint block' }, item.item_id)]),
        h('td', { className: 'wrap' }, item.category || '—'),
        h(
          'td',
          { className: 'wrap' },
          h(
            'ul',
            { className: 'chips' },
            fields.map((f) =>
              h('li', { className: filled.has(f) ? 'chip filled' : 'chip missing' }, `${filled.has(f) ? '已填' : '未填'}：${label(L.itemFields, f)}`),
            ),
          ),
        ),
        h('td', null, fmtTime(item.updated_at)),
        h('td', null, open),
      ]);
    });
    put($('items'), table('商品卡片', ['商品', '类目', '商家填写', '更新时间', '操作'], rows, true));
  }

  async function loadItemDetail(itemId, opener, focus = true) {
    const me = state.me;
    const box = $('item-detail');
    state.itemOpener = opener || null;
    box.hidden = false;
    box.dataset.itemId = itemId;
    loading(box);
    try {
      const card = await api('GET', shopPath(`console/items/${enc(itemId)}`));
      if (state.me !== me) return;
      renderItemDetail(card);
      if (focus) box.focus();
    } catch (err) {
      if (state.me !== me) return;
      loadFailed(box, `商品 ${itemId}`, err);
    }
  }

  function kvList(title, map) {
    const entries = Object.entries(map || {});
    if (!entries.length) return null;
    return [h('h5', null, title), h('dl', { className: 'stats' }, entries.map(([k, v]) => statRow(k, String(v))))];
  }

  function rowsTable(title, rows, skip, rename) {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return null;
    const keys = [];
    for (const row of list) for (const key of Object.keys(row)) if (!keys.includes(key) && !skip.includes(key)) keys.push(key);
    const body = list.map((row) => h('tr', null, keys.map((k) => h('td', null, row[k] === undefined || row[k] === null ? '—' : String(row[k])))));
    return table(title, keys.map((k) => rename[k] || k), body);
  }

  function renderItemDetail(card, message) {
    const box = $('item-detail');
    const titleId = uid('item-title');
    const close = h('button', { type: 'button', className: 'secondary' }, '收起');
    close.addEventListener('click', () => {
      box.hidden = true;
      delete box.dataset.itemId;
      put(box);
      if (state.itemOpener && state.itemOpener.isConnected) state.itemOpener.focus();
    });
    const platform = h('div', { className: 'item-platform' }, [
      h('h4', null, '平台字段（只读，由商品同步更新）'),
      h('dl', { className: 'stats' }, [
        statRow('标题', card.title),
        statRow('类目', card.category || '—'),
        statRow('卡片更新', fmtTime(card.updated_at)),
      ]),
      kvList('属性', card.attributes),
      rowsTable('SKU', card.skus, ['stock_source'], { sku_id: 'SKU' }),
      rowsTable('尺码表', card.size_chart, [], {}),
      card.detail_text ? [h('h5', null, '详情文字（含详情图识别的文字）'), h('p', { className: 'pre' }, card.detail_text)] : null,
    ]);
    const merchant = isMerchant() ? itemForm(card, message) : itemReadonly(card);
    box.setAttribute('role', 'region');
    box.setAttribute('aria-labelledby', titleId);
    put(box, h('div', { className: 'section-head' }, [h('h3', { id: titleId }, `商品卡片：${card.title}（${card.item_id}）`), close]), platform, merchant);
    const status = box.querySelector('.item-form .status');
    if (message && status) status.focus();
  }

  function itemReadonly(card) {
    const rows = Object.keys(L.itemFields).map((f) => {
      const value = card[f];
      const text = Array.isArray(value) ? (value.length ? value.join('；') : '未填') : hasText(value) ? value : '未填';
      return statRow(label(L.itemFields, f), text);
    });
    return h('div', { className: 'item-merchant' }, [h('h4', null, '商家填写'), h('dl', { className: 'stats' }, rows)]);
  }

  function itemForm(card, message) {
    const areas = {};
    const nodes = Object.keys(L.itemFields).map((f) => {
      const id = uid(`item-${f}`);
      const lim = ITEM_LIMITS[f] || {};
      const isList = f !== 'size_advice';
      const area = h('textarea', { id, rows: isList ? '3' : '2' });
      if (!isList && lim.max_chars) area.maxLength = lim.max_chars;
      area.value = isList ? (Array.isArray(card[f]) ? card[f].join('\n') : '') : card[f] || '';
      areas[f] = area;
      const hint = isList
        ? `每行一条，最多 ${lim.max_items} 条、每条 ${lim.max_chars} 个字；全部清空表示删掉。`
        : `最多 ${lim.max_chars} 个字；清空表示删掉。`;
      return field(id, label(L.itemFields, f), area, hint);
    });
    const save = h('button', { type: 'submit' }, '保存商家填写的字段');
    const sync = h('button', { type: 'button', className: 'secondary' }, '同步这件商品');
    const status = h('p', { className: 'status', role: 'status', tabindex: '-1' });
    const form = h('form', { className: 'item-form', novalidate: true }, [
      h('h4', null, '商家填写'),
      nodes,
      h('div', { className: 'actions tight' }, [save, sync]),
      status,
    ]);
    const ctx = { busy: false };
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const parsed = itemChanges(card, areas);
      if (parsed.error) {
        setStatus(status, parsed.error, 'error');
        areas[parsed.field].focus();
        return;
      }
      if (!Object.keys(parsed.body).length) {
        setStatus(status, '没有改动。', 'warn');
        return;
      }
      const me = state.me;
      clearError();
      await busy(ctx, [save, sync], async () => {
        try {
          const updated = await api('POST', shopPath(`console/items/${enc(card.item_id)}`), parsed.body);
          if (state.me !== me) return;
          const changed = (updated.changed || []).map((f) => label(L.itemFields, f));
          const text = changed.length ? `已保存：${changed.join('、')}。` : '没有改动。';
          renderItemDetail(updated, text);
          announce(text);
          stale('knowledge');
          refreshItemTable(me);
        } catch (err) {
          if (state.me !== me) return;
          reportFailure(status, '没有保存成功：', err);
        }
      });
    });
    sync.addEventListener('click', () => syncItems(card.item_id, [save, sync]));
    if (message) setStatus(status, message, 'ok');
    return form;
  }

  /** 商家填写的字段 → 有改动的字段（尺码建议清空为 null，列表清空为 null）。 */
  function itemChanges(card, areas) {
    const body = {};
    for (const [f, area] of Object.entries(areas)) {
      const lim = ITEM_LIMITS[f] || {};
      const name = label(L.itemFields, f);
      if (f === 'size_advice') {
        const text = area.value.trim();
        if (lim.max_chars && text.length > lim.max_chars) return { error: `${name}最多 ${lim.max_chars} 个字。`, field: f };
        const next = text || null;
        if (next !== (card.size_advice || null)) body.size_advice = next;
        continue;
      }
      const lines = linesOf(area.value);
      if (lim.max_items && lines.length > lim.max_items) return { error: `${name}最多 ${lim.max_items} 条。`, field: f };
      const long = lines.find((line) => lim.max_chars && line.length > lim.max_chars);
      if (long) return { error: `${name}每条最多 ${lim.max_chars} 个字：「${long.slice(0, 20)}…」`, field: f };
      if (!sameList(lines, Array.isArray(card[f]) ? card[f] : [])) body[f] = lines.length ? lines : null;
    }
    return { body };
  }

  async function refreshItemTable(me) {
    try {
      await fetchItems();
      if (state.me === me) renderItemTable();
    } catch (err) {
      /* 列表下次刷新时再取；详情已经是最新的 */
    }
  }

  function syncFailure(err) {
    const detail = err.data && typeof err.data.message === 'string' ? err.data.message : err.message;
    if (err.status === 503 && err.code === 'sync_unavailable') return `平台商品同步未接入：${detail}`;
    if (err.status === 503) return `平台商品接口暂时不可用，请稍后重试：${detail}`;
    return `同步没有成功：${err.message}`;
  }

  async function syncItems(itemId, controls) {
    const me = state.me;
    const box = $('sync-report');
    clearError();
    await busy(syncCtx, (controls || []).concat($('sync-all')), async () => {
      put(box, h('p', { className: 'muted' }, itemId ? `正在同步商品 ${itemId}…` : '正在同步全店商品…'));
      try {
        const report = await api('POST', shopPath('console/items/sync'), itemId ? { item_id: itemId } : {});
        if (state.me !== me) return;
        renderSyncReport(report);
        box.focus();
        announce('同步完成');
        stale('knowledge');
        await refreshItemTable(me);
        const detail = $('item-detail');
        const shown = detail.hidden ? null : detail.dataset.itemId;
        if (shown && state.me === me) await loadItemDetail(shown, state.itemOpener, false); // 平台字段可能变了
      } catch (err) {
        if (state.me !== me) return;
        const message = syncFailure(err);
        put(box, h('p', { className: 'status error' }, message));
        showError(message);
        box.focus();
      }
    });
  }

  function renderSyncReport(r) {
    const scope = r.scope === 'item' ? `商品 ${r.item_id}` : '全店';
    const rows = Object.entries(L.syncFields).map(([key, text]) => {
      const ids = Array.isArray(r[key]) ? r[key] : [];
      return statRow(text, ids.length ? `${ids.length} 件：${ids.join('、')}` : '0');
    });
    rows.push(statRow('没有写入（不属于本店 / 本平台）', String(r.rejected || 0)));
    const notes = [];
    if ((r.off_sale || []).length || (r.missing_on_platform || []).length) {
      notes.push(h('p', { className: 'hint warn-text' }, '已下架或平台上没有了的商品卡片保留、不删除；需要的话请人工处理。'));
    }
    if (r.ocr_failures) {
      notes.push(h('p', { className: 'hint warn-text' }, `详情图识别失败 ${r.ocr_failures} 张，其余字段照常同步，可以稍后再同步一次。`));
    }
    put($('sync-report'), h('p', { className: 'status ok' }, `同步完成（${scope}）· ${fmtTime(r.synced_at)}`), h('dl', { className: 'stats' }, rows), notes);
  }

  // ================================================================ 店铺设置（§9.3）

  async function loadConfig() {
    const me = state.me;
    loading($('config'));
    try {
      const view = await api('GET', shopPath('console/config'));
      if (state.me !== me) return;
      state.config = view;
      put($('config'), h('div', { className: 'config-list' }, CONFIG_FIELDS.map((f) => configRow(f))));
    } catch (err) {
      if (state.me !== me) return;
      loadFailed($('config'), '店铺设置', err);
    }
  }

  function formatConfig(fieldDef, value) {
    if (value === undefined || value === null) return '—';
    switch (fieldDef.kind) {
      case 'bool':
        return value ? '开' : '关';
      case 'intents':
        return value.length ? value.map((i) => label(L.intents, i)).join('、') : '（无）';
      case 'lines':
        return value.length ? value.join('、') : '（无）';
      case 'money':
        return `${value} 元`;
      case 'text':
        return value === '' ? '（空）' : String(value);
      default:
        return String(value);
    }
  }

  function configRow(fieldDef, message) {
    const view = state.config;
    const key = fieldDef.key;
    const effective = view.effective ? view.effective[key] : undefined;
    const fallback = view.defaults ? view.defaults[key] : undefined;
    const overridden = has(view.overrides || {}, key);
    const titleId = uid('cfg');
    const status = h('p', { className: 'status', role: 'status', tabindex: '-1' });
    const editable = fieldDef.editable && (view.editable || []).includes(key);
    const locked = (view.locked || []).includes(key) || !editable;
    const children = [
      h('div', { className: 'config-head' }, [
        h('h3', { id: titleId }, fieldDef.label),
        h('span', { className: overridden ? 'badge own' : 'badge' }, overridden ? '店铺设置' : '默认'),
      ]),
      fieldDef.hint ? h('p', { className: 'hint' }, fieldDef.hint) : null,
      h('dl', { className: 'config-values' }, [
        statRow('生效', formatConfig(fieldDef, effective)),
        statRow('默认', formatConfig(fieldDef, fallback)),
      ]),
    ];
    if (locked) {
      children.push(h('p', { className: 'hint locked' }, `不能在这里修改：${fieldDef.locked_reason || '这一项不开放给商家'}`));
    } else if (isMerchant()) {
      children.push(configEditor(fieldDef, effective, overridden, status, titleId));
    }
    children.push(status);
    const row = h('div', { className: 'config-row', role: 'group', 'aria-labelledby': titleId, 'data-key': key }, children);
    if (message) setStatus(status, message, 'ok');
    return row;
  }

  function configEditor(fieldDef, value, overridden, status, titleId) {
    const id = uid(`cfg-${fieldDef.key}`);
    let control;
    let read;
    let focusTarget;
    const editLabel = `修改${fieldDef.label}`;
    switch (fieldDef.kind) {
      case 'select': {
        const select = h('select', { id }, (fieldDef.options || []).map((o) => h('option', { value: o }, o)));
        select.value = value;
        control = field(id, editLabel, select);
        focusTarget = select;
        read = () => ({ value: select.value });
        break;
      }
      case 'int': {
        const input = h('input', { id, type: 'number', inputmode: 'numeric', step: '1', min: String(fieldDef.min), max: String(fieldDef.max) });
        input.value = value === undefined || value === null ? '' : String(value);
        control = field(id, editLabel, input);
        focusTarget = input;
        read = () => {
          const n = Number(input.value);
          if (!input.value.trim() || !Number.isInteger(n) || n < fieldDef.min || n > fieldDef.max) {
            return { error: `${fieldDef.label}要是 ${fieldDef.min}–${fieldDef.max} 之间的整数。` };
          }
          return { value: n };
        };
        break;
      }
      case 'money': {
        const input = h('input', { id, type: 'number', inputmode: 'decimal', step: '0.01', min: '0' });
        input.value = value === undefined || value === null ? '' : String(value);
        control = field(id, editLabel, input);
        focusTarget = input;
        read = () => {
          const text = input.value.trim();
          const n = Number(text);
          if (!text || !Number.isFinite(n) || n < 0) return { error: `${fieldDef.label}要是不小于 0 的金额。` };
          return { value: text };
        };
        break;
      }
      case 'lines': {
        const area = h('textarea', { id, rows: '3' });
        area.value = Array.isArray(value) ? value.join('\n') : '';
        control = field(id, editLabel, area, `每行一个，最多 ${fieldDef.max_items} 个、每个 ${fieldDef.max_chars} 个字。`);
        focusTarget = area;
        read = () => {
          const lines = linesOf(area.value);
          if (lines.length > fieldDef.max_items) return { error: `${fieldDef.label}最多 ${fieldDef.max_items} 个。` };
          const long = lines.find((line) => line.length > fieldDef.max_chars);
          if (long) return { error: `${fieldDef.label}每个最多 ${fieldDef.max_chars} 个字：「${long.slice(0, 20)}…」` };
          return { value: lines };
        };
        break;
      }
      case 'intents': {
        const current = new Set(Array.isArray(value) ? value : []);
        const boxes = Object.entries(L.intents).map(([intent, text]) => {
          const boxId = `${id}-${intent}`;
          const input = h('input', { type: 'checkbox', id: boxId, value: intent });
          input.checked = current.has(intent);
          return h('div', { className: 'check' }, [input, h('label', { for: boxId }, text)]);
        });
        control = h('fieldset', { className: 'intent-grid' }, [h('legend', null, '关闭这些问题类型（不给推荐）'), boxes]);
        focusTarget = control.querySelector('input');
        read = () => ({ value: Array.from(control.querySelectorAll('input:checked'), (i) => i.value) });
        break;
      }
      default: {
        const input = h('input', { id, type: 'text', maxlength: String(fieldDef.max_chars || 64), autocomplete: 'off' });
        input.value = value || '';
        control = field(id, editLabel, input);
        focusTarget = input;
        read = () => {
          const text = input.value.trim();
          if (fieldDef.max_chars && text.length > fieldDef.max_chars) return { error: `${fieldDef.label}最多 ${fieldDef.max_chars} 个字。` };
          return { value: text };
        };
      }
    }
    const save = h('button', { type: 'submit', 'aria-describedby': titleId }, '保存');
    const reset = overridden ? h('button', { type: 'button', className: 'secondary', 'aria-describedby': titleId }, '恢复默认') : null;
    const form = h('form', { className: 'config-form', novalidate: true }, [control, h('div', { className: 'actions tight' }, [save, reset])]);
    const ctx = { busy: false };
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const parsed = read();
      if (parsed.error) {
        setStatus(status, parsed.error, 'error');
        focusTarget.focus();
        return;
      }
      saveConfig(fieldDef, parsed.value, ctx, [save, reset], status);
    });
    if (reset) reset.addEventListener('click', () => saveConfig(fieldDef, null, ctx, [save, reset], status));
    return form;
  }

  async function saveConfig(fieldDef, value, ctx, controls, status) {
    const me = state.me;
    clearError();
    await busy(ctx, controls, async () => {
      try {
        const view = await api('POST', shopPath('console/config'), { changes: { [fieldDef.key]: value } });
        if (state.me !== me) return;
        state.config = view;
        const now = formatConfig(fieldDef, view.effective ? view.effective[fieldDef.key] : undefined);
        let message = '没有改动。';
        if ((view.removed || []).includes(fieldDef.key)) message = `已恢复默认：${fieldDef.label}（现在是 ${now}）。`;
        else if ((view.changed || []).includes(fieldDef.key)) message = `已保存：${fieldDef.label} → ${now}。`;
        const old = $('config').querySelector(`[data-key="${fieldDef.key}"]`);
        const fresh = configRow(fieldDef, message);
        if (old) old.replaceWith(fresh);
        fresh.querySelector('.status').focus();
        announce(message);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有保存成功：', err);
      }
    });
  }

  // ================================================================ 上线节奏（§13.4）

  async function loadRollout() {
    const me = state.me;
    loading($('rollout'));
    put($('stage-controls'));
    try {
      const status = await api('GET', shopPath('rollout/status'));
      if (state.me !== me) return;
      renderRollout(status);
      renderStageControls(status);
    } catch (err) {
      if (state.me !== me) return;
      loadFailed($('rollout'), '上线状态', err);
    }
  }

  function checkLabel(name) {
    let rest = String(name);
    let prefix = '';
    const scoped = /^intent:([a-z_]+):(.+)$/.exec(rest); // 全自动阶段：按类型的门槛
    if (scoped) {
      prefix = `${label(L.intents, scoped[1])} · `;
      rest = scoped[2];
    }
    if (has(L.gateChecks, rest)) return prefix + L.gateChecks[rest];
    const period = /^period_(\d+)\.(.+)$/.exec(rest);
    if (period) return `${prefix}第 ${period[1]} 周${label(L.gateChecks, period[2])}`;
    return prefix + rest;
  }

  function checkValue(name, value) {
    if (value === null || value === undefined) return '—';
    return /rate$/.test(String(name)) ? pct(value) : num(value);
  }

  function gateTable(gate, title) {
    const checks = Array.isArray(gate.checks) ? gate.checks : [];
    const unchecked = Array.isArray(gate.unchecked) ? gate.unchecked : [];
    const rows = checks.map((c) =>
      h('tr', null, [
        h('th', { scope: 'row', className: 'wrap' }, checkLabel(c.name)),
        h('td', null, checkValue(c.name, c.value)),
        h('td', null, checkValue(c.name, c.threshold)),
        h('td', null, h('span', { className: c.passed ? 'badge ok' : 'badge bad' }, c.passed ? '达到' : '没达到')),
        h('td', { className: 'wrap' }, c.detail || ''),
      ]),
    );
    return h('div', { className: 'gate' }, [
      h('p', { className: 'gate-title' }, [
        h('strong', null, title),
        ' ',
        h('span', { className: gate.passed ? 'badge ok' : 'badge bad' }, gate.passed ? '已检查的条件都达到' : '没达到'),
      ]),
      table(`${title}：各项条件`, ['条件', '当前', '门槛', '结果', '说明'], rows, true),
      unchecked.length ? h('p', { className: 'hint' }, `没有检查（需要人工确认）：${unchecked.map((n) => checkLabel(n)).join('、')}`) : null,
    ]);
  }

  function currentGates(s) {
    if (s.stage === 'shadow') {
      return [{ title: '影子模式 → 智能辅助（§13.4：测试集和影子对比都达到 §13.2 门槛）', gate: s.shadow_gate }];
    }
    if (s.stage === 'assist') {
      return [{ title: '智能辅助 → 下一阶段（§13.4：连续 2 周采纳率 ≥ 60%，平台违规为 0）', gate: s.assist_gate }];
    }
    return Object.entries(s.intent_gates || {}).map(([intent, gate]) => ({
      title: `全自动 · ${label(L.intents, intent)}（连续 4 周原样采纳率 ≥ 90%，没有投诉）`,
      gate,
    }));
  }

  function renderRollout(s) {
    const stageText = s.stage_label || label(L.stages, s.stage);
    const blocking = Array.isArray(s.blocking) ? s.blocking : [];
    const unchecked = Array.isArray(s.unchecked) ? s.unchecked : [];
    const nodes = [
      h('div', { className: `stage-card stage-${s.stage}` }, [
        h('p', { className: 'stage-now' }, ['当前阶段：', h('strong', { id: 'stage-label' }, stageText)]),
        h('p', { className: 'advice' }, `建议：${s.advice_text || s.advice}`),
        h('p', { className: 'hint' }, `统计截至 ${fmtTime(s.as_of)}`),
      ]),
      h('h3', null, '下一道门槛'),
      blocking.length
        ? h('div', { className: 'warnings' }, [
            h('p', { className: 'warnings-title' }, '还没达到的条件：'),
            h('ul', null, blocking.map((name) => h('li', null, checkLabel(name)))),
          ])
        : h('p', { className: 'status ok' }, '当前阶段已检查的条件都达到了。'),
      unchecked.length
        ? h('div', { className: 'unchecked' }, [
            h('p', null, '需要人工确认的条件（没有数据来源，切换时由复核人员确认）：'),
            h('ul', null, unchecked.map((u) => h('li', null, u.description || u.name))),
          ])
        : null,
      currentGates(s).map((g) => (g.gate ? gateTable(g.gate, g.title) : null)),
    ];

    const sh = s.shadow || {};
    const q = s.shadow_quality || {};
    const detailBox = h('div', { className: 'shadow-detail' });
    const details = h('details', { className: 'subpanel' }, [h('summary', null, '逐组明细（推荐编号和相似度，没有文字）'), detailBox]);
    details.addEventListener('toggle', () => {
      if (details.open) loadShadowDetail(detailBox);
    });
    nodes.push(
      h('h3', null, '影子对比'),
      h('dl', { className: 'stats' }, [
        statRow('对比窗口', fmtRange(sh.since, sh.until)),
        statRow('影子推荐组', `${num(sh.groups)}（客服有回复、可对比 ${num(sh.compared)}，客服没回复 ${num(sh.no_reply)}）`),
        statRow('字符重合度', `平均 ${pct(sh.mean_similarity)} · 中位数 ${pct(sh.median_similarity)}`),
        statRow('重合度达到阈值的比例', `${pct(sh.match_rate)}（阈值 ${pct(sh.match_threshold)}；只作参考，不是门槛）`),
        statRow('影子样本复核', `${label(L.gateStatus, q.status)}：${q.detail || '—'}`),
        statRow('事实正确 / 合规满分率', `${pct(q.factual_full_rate)} / ${pct(q.compliance_full_rate)}（门槛 ≥ ${pct(q.min_full_rate)}）`),
      ]),
      details,
    );

    const weekly = Array.isArray(s.weekly) ? s.weekly : [];
    nodes.push(
      h('h3', null, '每周采纳率（按推荐组）'),
      weekly.length
        ? table(
            '每周采纳率',
            ['周期', '日期', '推荐组', '采纳率', '原样采纳率', '影子推荐（不计入）'],
            weekly.map((w) =>
              h('tr', null, [
                h('th', { scope: 'row' }, w.period),
                h('td', null, fmtRange(w.since, w.until)),
                h('td', null, num(w.groups)),
                h('td', null, pct(w.group_adoption_rate)),
                h('td', null, pct(w.group_as_is_rate)),
                h('td', null, num(w.shadow_excluded)),
              ]),
            ),
            true,
          )
        : h('p', { className: 'muted' }, '还没有数据。'),
    );

    const lq = s.latest_quality;
    nodes.push(h('h3', null, '最近一次每周复核'));
    if (lq) {
      const g = lq.gate || {};
      nodes.push(
        h('dl', { className: 'stats' }, [
          statRow('周期', lq.period || '—'),
          statRow('样本', `抽中 ${num(lq.sampled)}，已打分 ${num(lq.scored)}，跳过 ${num(lq.skipped)}，待复核 ${num(lq.pending)}`),
          statRow('§13.2 门槛', `${label(L.gateStatus, g.status)}：${g.detail || '—'}`),
        ]),
      );
    } else {
      nodes.push(h('p', { className: 'muted' }, '最近几周没有复核样本。'));
    }
    put($('rollout'), nodes);
  }

  async function loadShadowDetail(box) {
    const me = state.me;
    loading(box);
    try {
      const view = await api('GET', shopPath('rollout/shadow'));
      if (state.me !== me) return;
      const items = Array.isArray(view.items) ? view.items : [];
      if (!items.length) {
        put(box, h('p', { className: 'muted' }, `${fmtRange(view.since, view.until)} 没有影子推荐。`));
        return;
      }
      const rows = items.map((i) =>
        h('tr', null, [
          h('td', null, i.suggestion_id === null || i.suggestion_id === undefined ? '—' : `#${i.suggestion_id}`),
          h('td', null, label(L.intents, i.intent)),
          h('td', null, i.similarity === null || i.similarity === undefined ? '客服没有回复' : pct(i.similarity)),
        ]),
      );
      put(box, table(`影子对比明细（${fmtRange(view.since, view.until)}）`, ['推荐', '问题类型', '和客服回复的重合度'], rows));
    } catch (err) {
      if (state.me !== me) return;
      loadFailed(box, '影子对比明细', err);
    }
  }

  function renderStageControls(s) {
    const box = $('stage-controls');
    const role = state.me.role;
    const canShadow = inGroup('stage_shadow', role);
    const canAssist = inGroup('stage_assist', role);
    if (!canShadow && !canAssist) {
      put(box, h('p', { className: 'muted' }, '切换阶段由商家（只能切回影子模式）或复核人员操作。'));
      return;
    }
    const ctx = { busy: false };
    const status = h('p', { className: 'status', role: 'status', tabindex: '-1' });
    const panel = h('div', { className: 'entry-panel' });
    const actions = h('div', { className: 'actions tight' });
    const notes = [];
    if (s.stage !== 'shadow' && canShadow) {
      const text = canAssist ? '切到影子模式' : '切回影子模式';
      const button = h('button', { type: 'button', className: 'secondary' }, text);
      button.addEventListener('click', () => confirmShadow({ ctx, actions, panel, status, opener: button, text }));
      actions.append(button);
    }
    if (s.stage === 'shadow' && canAssist) {
      const button = h('button', { type: 'button' }, '切到智能辅助');
      button.addEventListener('click', () =>
        postStage({ stage: 'assist' }, { ctx, actions, panel, status, controls: [button] }),
      );
      actions.append(button);
    }
    if (s.stage === 'shadow' && !canAssist) {
      notes.push(h('p', { className: 'hint' }, '已在影子模式：推荐只在后台生成、不展示。恢复智能辅助要复核人员确认 §13.4 的门槛。'));
    } else if (!canAssist) {
      notes.push(h('p', { className: 'hint' }, '商家只能切回影子模式；切到智能辅助由复核人员确认门槛后操作。'));
    }
    if (s.stage === 'auto') {
      notes.push(h('p', { className: 'hint' }, '店铺已开全自动（平台准入后由我们开启），这里只能切回影子模式。'));
    }
    put(box, notes, actions, panel, status);
    if (state.stageMessage) {
      setStatus(status, state.stageMessage, 'ok');
      state.stageMessage = null;
      status.focus();
    }
  }

  function confirmShadow(ui) {
    const confirm = h('button', { type: 'button', className: 'danger' }, `确认${ui.text}`);
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const close = () => {
      ui.panel.replaceChildren();
      ui.actions.hidden = false;
      ui.opener.focus();
    };
    cancel.addEventListener('click', close);
    const box = h('div', { className: 'confirm' }, [
      h('p', null, '切到影子模式后，推荐只在后台生成、不再展示给客服，客服按平时的方式回复（§13.4）。恢复智能辅助要复核人员确认门槛。'),
      h('div', { className: 'actions tight' }, [confirm, cancel]),
    ]);
    box.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
      }
    });
    confirm.addEventListener('click', () => postStage({ stage: 'shadow' }, { ...ui, controls: [confirm, cancel] }));
    ui.actions.hidden = true;
    ui.panel.replaceChildren(box);
    confirm.focus();
  }

  async function postStage(body, ui) {
    const me = state.me;
    clearError();
    await busy(ui.ctx, ui.controls, async () => {
      try {
        const change = await api('POST', shopPath('rollout/stage'), body);
        if (state.me !== me) return;
        const previous = label(L.stages, change.previous);
        state.stageMessage = change.changed
          ? `已切换：${previous} → ${change.stage_label}${change.forced ? '（门槛没达到，强制切换，原因已记入审计）' : ''}。`
          : `已经是${change.stage_label}，没有改动。`;
        announce(state.stageMessage);
        stale('config'); // 设置页里影子模式的开关跟着变
        await loadRollout();
      } catch (err) {
        if (state.me !== me) return;
        if (err.status === 409 && err.code === 'gate_not_passed' && err.data && err.data.gate) {
          showGateFailure(err.data, ui);
          return;
        }
        reportFailure(ui.status, '没有切换：', err);
      }
    });
  }

  /** 影子 → 智能辅助的门槛没过（409）：显示门槛结果，复核人员可以写明原因后强制切换。 */
  function showGateFailure(data, ui) {
    const titleId = uid('gate-fail');
    const noteId = uid('force-note');
    const note = h('textarea', { id: noteId, rows: '2', maxlength: String(LIMITS.note) });
    const force = h('button', { type: 'submit', className: 'danger' }, '强制切到智能辅助');
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const close = () => {
      ui.panel.replaceChildren();
      ui.actions.hidden = false;
      const first = ui.actions.querySelector('button');
      if (first) first.focus();
    };
    cancel.addEventListener('click', close);
    const form = h('form', { className: 'entry-form', novalidate: true }, [
      field(noteId, '强制切换的原因（必填，记入审计）', note, '比如：测试集已人工确认达标、影子样本不足但已抽查。'),
      h('div', { className: 'actions tight' }, [force, cancel]),
    ]);
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = note.value.trim();
      if (!text) {
        setStatus(ui.status, '强制切换要写明原因。', 'error');
        note.focus();
        return;
      }
      postStage({ stage: 'assist', force: true, note: text }, { ...ui, controls: [force, cancel] });
    });
    const box = h('div', { className: 'gate-failure', role: 'group', 'aria-labelledby': titleId }, [
      h('p', { id: titleId, className: 'warnings-title', tabindex: '-1' }, `门槛没过，没有切换：${data.message || ''}`),
      gateTable(data.gate, '影子模式 → 智能辅助'),
      form,
    ]);
    ui.actions.hidden = true;
    ui.panel.replaceChildren(box);
    setStatus(ui.status, '', '');
    $(titleId).focus();
    announce('门槛没过，没有切换');
  }

  // ================================================================ 每周复核（§13.5，按 §13.2 打分）

  const sampleCtx = { busy: false };
  const queueCtx = { busy: false };

  function periodValue() {
    const text = $('review-period').value.trim();
    if (!text) return { value: null };
    if (!PERIOD_RE.test(text)) return { error: '周期要写成 ISO 周，如 2026-W40。' };
    return { value: text };
  }

  function loadReviewTab() {
    const me = state.me;
    loadQualityReport(null, true);
    if (state.me === me && !$('review-queue').childNodes.length) {
      put($('review-queue'), h('p', { className: 'muted' }, '点「加载复核列表」查看样本内容（会记录访问）。'));
    }
  }

  async function loadQualityReport(period, fillPeriod) {
    const me = state.me;
    const box = $('review-report');
    loading(box);
    try {
      const report = await api('GET', shopPath('quality/report', { period }));
      if (state.me !== me) return;
      if (fillPeriod && report.period && !$('review-period').value) $('review-period').value = report.period;
      renderQualityReport(report);
    } catch (err) {
      if (state.me !== me) return;
      loadFailed(box, '复核报表', err);
    }
  }

  function renderQualityReport(r) {
    const gate = r.gate || {};
    const sampled = r.sampled || 0;
    const done = (r.scored || 0) + (r.skipped || 0);
    const dims = r.dimensions || {};
    const scores = Array.from({ length: FULL_SCORE + 1 }, (_, score) => score);
    const dimRows = Object.entries(L.dimensions).map(([key, text]) => {
      const d = dims[key] || {};
      const dist = d.distribution || {};
      return h('tr', null, [
        h('th', { scope: 'row' }, text),
        h('td', null, num(d.scored)),
        h('td', null, pct(d.full_rate)),
        h('td', null, num(d.mean)),
        scores.map((score) => h('td', null, num(dist[score] || 0))),
      ]);
    });
    const byIntent = Object.values(r.by_intent || {});
    const intentRows = byIntent.map((s) => {
      const d = s.dimensions || {};
      return h('tr', null, [
        h('th', { scope: 'row' }, s.label || label(L.intents, s.intent)),
        h('td', null, num(s.sampled)),
        h('td', null, num(s.scored)),
        h('td', null, num(s.skipped)),
        h('td', null, num(s.pending)),
        h('td', null, pct(d.factual && d.factual.full_rate)),
        h('td', null, pct(d.compliance && d.compliance.full_rate)),
      ]);
    });
    const gateClass = gate.status === 'passed' ? 'badge ok' : gate.status === 'failed' ? 'badge bad' : 'badge';
    put(
      $('review-report'),
      h('p', { className: 'summary' }, `周期 ${r.period}（${fmtRange(r.since, r.until)}）`),
      h('progress', { max: String(Math.max(1, sampled)), value: String(done), 'aria-label': '复核进度' }),
      h(
        'p',
        { className: 'progress-text', id: 'review-progress-text' },
        sampled
          ? `已复核 ${done} / ${sampled}（打分 ${num(r.scored)}，跳过 ${num(r.skipped)}，待复核 ${num(r.pending)}）`
          : '这个周期还没有抽样。',
      ),
      h('div', { className: `gate-box gate-${gate.status}` }, [
        h('p', null, [
          h('strong', null, `§13.2 上线门槛：事实正确和合规的满分率都 ≥ ${pct(gate.min_full_rate)}`),
          ' ',
          h('span', { className: gateClass, id: 'quality-gate-status' }, label(L.gateStatus, gate.status)),
        ]),
        h('p', { className: 'hint' }, `${gate.detail || ''}（已打分至少 ${num(gate.min_scored)} 条才算数；跳过的不计入满分率）`),
        h('dl', { className: 'stats' }, [
          statRow('事实正确满分率', `${pct(gate.factual_full_rate)}（${gate.factual_ok ? '达到' : '没达到'}）`),
          statRow('合规满分率', `${pct(gate.compliance_full_rate)}（${gate.compliance_ok ? '达到' : '没达到'}）`),
        ]),
      ]),
      table('各维度得分（§13.2）', ['维度', '已打分', '满分率', '平均分', ...scores.map((s) => `${s} 分`)], dimRows),
      intentRows.length
        ? table('按问题类型', ['问题类型', '抽中', '已打分', '跳过', '待复核', '事实正确满分率', '合规满分率'], intentRows)
        : null,
    );
  }

  async function sample() {
    if (!state.me) return;
    const p = periodValue();
    const status = $('review-status');
    if (p.error) {
      setStatus(status, p.error, 'error');
      $('review-period').focus();
      return;
    }
    const size = Number($('review-size').value || LIMITS.sample_size);
    if (!Number.isInteger(size) || size < 1 || size > LIMITS.sample_size_max) {
      setStatus(status, `抽样条数要在 1–${LIMITS.sample_size_max} 之间。`, 'error');
      $('review-size').focus();
      return;
    }
    const body = { size };
    if (p.value) body.period = p.value;
    const me = state.me;
    clearError();
    await busy(sampleCtx, [$('review-sample'), $('review-report-button')], async () => {
      setStatus(status, '正在抽样…', 'muted');
      try {
        const r = await api('POST', shopPath('quality/sample'), body);
        if (state.me !== me) return;
        $('review-period').value = r.period;
        let text;
        if (r.added) text = `周期 ${r.period}：可抽 ${r.eligible} 条推荐，本次新抽 ${r.added} 条，共 ${r.total} 条样本。`;
        else if (r.total) text = `周期 ${r.period}：已有 ${r.total} 条样本，没有新抽（重复抽样只补不换）。`;
        else text = `周期 ${r.period} 没有可抽的推荐。`;
        setStatus(status, text, r.total ? 'ok' : 'warn');
        announce(text);
        stale('rollout');
        await loadQualityReport(r.period);
        if (r.total && state.me === me) await loadQueue(r.period);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有抽样成功：', err);
      }
    });
  }

  async function loadQueue(periodOverride) {
    if (!state.me) return;
    const p = periodOverride ? { value: periodOverride } : periodValue();
    if (p.error) {
      setStatus($('review-status'), p.error, 'error');
      $('review-period').focus();
      return;
    }
    const me = state.me;
    const box = $('review-queue');
    clearError();
    await busy(queueCtx, [$('review-load')], async () => {
      loading(box);
      try {
        const list = await api('GET', shopPath('quality/reviews', { period: p.value, status: $('review-filter').value }));
        if (state.me !== me) return;
        state.review = { period: list.period, items: Array.isArray(list.items) ? list.items : [] };
        renderQueue(list);
        if (!periodOverride) await loadQualityReport(list.period); // 报表和列表看同一个周期
      } catch (err) {
        if (state.me !== me) return;
        loadFailed(box, '复核列表', err);
      }
    });
  }

  function renderQueue(list) {
    const items = state.review.items;
    const scope = list.status ? ` · ${label(L.reviewStatus, list.status)}` : '';
    const head = h('p', { className: 'summary', id: 'review-queue-summary' }, `周期 ${list.period}${scope}：${items.length} 条样本。本次查看已记录访问日志。`);
    if (!items.length) {
      put($('review-queue'), head, h('p', { className: 'muted' }, '没有样本。先选周期、点「抽样」。'));
      return;
    }
    put($('review-queue'), head, h('ol', { className: 'review-list' }, items.map((item) => h('li', null, reviewCard(item)))));
  }

  function reviewCard(item, message) {
    const titleId = uid('rv');
    const s = item.suggestion;
    const status = h('p', { className: 'status', role: 'status', tabindex: '-1' });
    const content = s
      ? [
          h('p', { className: 'rv-label' }, '买家问题（已打码）'),
          h('blockquote', { className: 'buyer' }, s.question_text || '（没有记录买家问题）'),
          h('p', { className: 'rv-label' }, '推荐回答'),
          h('p', { className: 'card-text' }, s.text),
          s.edited_text ? [h('p', { className: 'rv-label' }, '客服编辑后发出的文字'), h('p', { className: 'card-text edited' }, s.edited_text)] : null,
          h('div', { className: 'evidence' }, [
            h('span', { className: 'evidence-title' }, '依据：'),
            s.evidence_ids && s.evidence_ids.length
              ? h('ul', { className: 'chips' }, s.evidence_ids.map((e) => h('li', { className: 'chip' }, e)))
              : h('span', { className: 'muted' }, '无'),
          ]),
          s.reject_reason ? h('p', { className: 'hint' }, `客服标记不合适：${label(L.rejectReasons, s.reject_reason)}`) : null,
        ]
      : [h('p', { className: 'warn-text' }, '推荐已不存在，看不到内容；建议跳过。')];
    const reviewed =
      item.status !== 'pending' && item.reviewed_at
        ? h(
            'p',
            { className: 'hint' },
            `${label(L.reviewStatus, item.status)} · ${item.reviewer || '—'} · ${fmtTime(item.reviewed_at)}${item.note ? ` · 备注：${item.note}` : ''}`,
          )
        : null;
    const article = h('article', { className: `review-item rv-${item.status}`, 'aria-labelledby': titleId, 'data-review': String(item.id) }, [
      h('h4', { id: titleId }, `样本 #${item.id} · 推荐 #${item.suggestion_id}`),
      h('p', { className: 'card-meta' }, [
        h('span', { className: `badge rv-${item.status}` }, label(L.reviewStatus, item.status)),
        h('span', { className: 'badge' }, item.intent_label || label(L.intents, item.intent)),
        s ? h('span', { className: 'badge' }, label(L.kinds, s.kind)) : null,
        s ? h('span', { className: 'badge' }, label(L.suggestionStatus, s.status)) : null,
        s ? h('span', { className: 'badge' }, `生成于 ${fmtTime(s.created_at)}`) : null,
      ]),
      content,
      reviewed,
      scoreForm(item, status, titleId),
      status,
    ]);
    if (message) setStatus(status, message, 'ok');
    return article;
  }

  function scoreForm(item, status, titleId) {
    const name = uid('score');
    const scores = item.scores || {};
    const groups = Object.entries(L.dimensions).map(([dim, text]) => {
      const guide = L.scoreGuide[dim] || [];
      const radios = [];
      for (let score = 0; score <= FULL_SCORE; score += 1) {
        const id = `${name}-${dim}-${score}`;
        const input = h('input', { type: 'radio', name: `${name}-${dim}`, id, value: String(score) });
        if (scores[dim] === score) input.checked = true;
        radios.push(h('div', { className: 'radio' }, [input, h('label', { for: id }, `${score} 分 · ${guide[score] || ''}`)]));
      }
      return h('fieldset', { className: 'score-dim', 'data-dim': dim }, [h('legend', null, text), radios]);
    });
    const noteId = `${name}-note`;
    const note = h('textarea', { id: noteId, rows: '2', maxlength: String(LIMITS.note) });
    note.value = item.note || '';
    const submit = h('button', { type: 'submit', 'aria-describedby': titleId }, item.status === 'scored' ? '修改评分' : '提交评分');
    const skip = h('button', { type: 'button', className: 'secondary', 'aria-describedby': titleId }, '跳过');
    const form = h('form', { className: 'score-form', novalidate: true }, [
      h('div', { className: 'score-grid' }, groups),
      field(noteId, '备注（选填，保存前会打码）', note),
      h('div', { className: 'actions tight' }, [submit, skip]),
    ]);
    const ctx = { busy: false };
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const picked = {};
      for (const group of form.querySelectorAll('fieldset[data-dim]')) {
        const checked = group.querySelector('input:checked');
        if (!checked) {
          setStatus(status, `请给「${group.querySelector('legend').textContent}」打分（0–${FULL_SCORE}）。`, 'error');
          group.querySelector('input').focus();
          return;
        }
        picked[group.dataset.dim] = Number(checked.value);
      }
      submitScore(item, { status: 'scored', scores: picked, note: note.value }, ctx, [submit, skip], status);
    });
    skip.addEventListener('click', () => submitScore(item, { status: 'skipped', note: note.value }, ctx, [submit, skip], status));
    return form;
  }

  async function submitScore(item, result, ctx, controls, status) {
    const body = { status: result.status };
    if (result.scores) body.scores = result.scores;
    const note = String(result.note || '').trim();
    if (note.length > LIMITS.note) {
      setStatus(status, `备注最多 ${LIMITS.note} 个字。`, 'error');
      return;
    }
    if (note) body.note = note;
    const me = state.me;
    clearError();
    await busy(ctx, controls, async () => {
      try {
        const updated = await api('POST', shopPath(`quality/reviews/${enc(String(item.id))}`), body);
        if (state.me !== me) return;
        const merged = {
          ...item,
          status: updated.status,
          scores: updated.scores,
          reviewer: updated.reviewer,
          note: updated.note,
          reviewed_at: updated.reviewed_at,
        };
        state.review.items = state.review.items.map((i) => (i.id === item.id ? merged : i));
        const message = updated.status === 'scored' ? `已记录样本 #${item.id} 的评分。` : `已跳过样本 #${item.id}。`;
        const old = $('review-queue').querySelector(`[data-review="${item.id}"]`);
        const fresh = reviewCard(merged, message);
        if (old) old.replaceWith(fresh);
        fresh.querySelector('.status').focus();
        announce(message);
        stale('rollout');
        await loadQualityReport(state.review.period || item.period);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有记录成功：', err);
      }
    });
  }

  // ================================================================ 报表（§12.4）

  function sinceParam() {
    const days = Number($('range').value);
    if (!days) return null;
    return new Date(Date.now() - days * 86400000).toISOString();
  }

  async function loadReports() {
    const me = state.me;
    loading($('adoption'));
    loading($('qa-report'));
    const since = sinceParam();
    const [adoption, qa] = await Promise.allSettled([
      api('GET', shopPath('reports/adoption', { since })),
      api('GET', shopPath('reports/qa', { since })),
    ]);
    if (state.me !== me) return;
    if (adoption.status === 'fulfilled') renderAdoption(adoption.value);
    else loadFailed($('adoption'), '采纳率报表', adoption.reason);
    if (qa.status === 'fulfilled') renderQa(qa.value);
    else loadFailed($('qa-report'), '质检报表', qa.reason);
  }

  function reasonSummary(stats) {
    const parts = Object.entries(stats.reject_reasons || {})
      .filter(([, count]) => count > 0)
      .map(([reason, count]) => `${label(L.rejectReasons, reason)} ${count}`);
    return parts.length ? parts.join('、') : '—';
  }

  function renderAdoption(report) {
    const o = report && report.overall;
    if (!o || !o.total) {
      put($('adoption'), h('p', { className: 'muted' }, '这段时间还没有推荐记录。'));
      return;
    }
    const rate = o.group_adoption_rate;
    const reached = typeof rate === 'number' && rate >= ADOPTION_TARGET;
    const byIntent = Object.values(report.by_intent || {});
    put(
      $('adoption'),
      h('p', { className: 'headline' }, [
        h('strong', { className: reached ? 'good' : 'bad' }, pct(rate)),
        ` 采纳率（按买家消息算；Phase 0 目标 ≥ ${pct(ADOPTION_TARGET)}，§15）`,
      ]),
      h('dl', { className: 'stats' }, [
        statRow('推荐', `${o.total} 条 / ${o.groups} 组消息`),
        statRow('原样发送', `${o.sent}（${pct(o.as_is_rate)}）`),
        statRow('编辑后发送', `${o.edited_sent}（${pct(o.edited_rate)}）`),
        statRow('不合适', `${o.rejected}（${pct(o.reject_rate)}）`),
        statRow('已过期 / 未操作', `${o.expired} / ${o.shown}`),
        statRow('弃用原因', reasonSummary(o)),
      ]),
      byIntent.length
        ? table(
            '按问题类型',
            ['问题类型', '消息', '采纳率', '原样', '编辑后', '不合适', '弃用原因'],
            byIntent.map((s) =>
              h('tr', null, [
                h('th', { scope: 'row' }, s.label || label(L.intents, s.intent)),
                h('td', null, num(s.groups)),
                h('td', null, pct(s.group_adoption_rate)),
                h('td', null, num(s.sent)),
                h('td', null, num(s.edited_sent)),
                h('td', null, num(s.rejected)),
                h('td', null, reasonSummary(s)),
              ]),
            ),
          )
        : null,
      report.shadow_excluded ? h('p', { className: 'hint' }, `影子模式生成的 ${report.shadow_excluded} 条推荐不计入。`) : null,
    );
  }

  function countTable(caption, head, counts, labels) {
    const entries = Object.entries(counts || {}).filter(([, n]) => n > 0);
    if (!entries.length) return null;
    return table(
      caption,
      [head, '次数'],
      entries.map(([key, n]) => h('tr', null, [h('th', { scope: 'row' }, labels(key)), h('td', null, num(n))])),
    );
  }

  function degradedLabel(code) {
    const text = String(code);
    if (has(L.degraded, text)) return L.degraded[text];
    const [kind, ...rest] = text.split(':');
    if (kind === 'tool' && L.degraded.tool) return `${L.degraded.tool}（${rest.join(':')}）`;
    return text;
  }

  function renderQa(r) {
    if (!r || !r.messages) {
      put($('qa-report'), h('p', { className: 'muted' }, '这段时间还没有处理过买家消息。'));
      return;
    }
    const p95 = r.p95_within_target === true ? '达到' : r.p95_within_target === false ? '没达到' : '没有数据';
    put(
      $('qa-report'),
      h('dl', { className: 'stats' }, [
        statRow('处理的买家消息', num(r.messages)),
        statRow('质检拦截的草稿', `${num(r.qa_failed)} / ${num(r.qa_checked)}（${pct(r.interception_rate)}）`),
        statRow('有拦截的消息', `${num(r.messages_intercepted)} / ${num(r.messages_checked)}（${pct(r.message_interception_rate)}）`),
        statRow('有降级的消息', `${num(r.degraded_messages)}（${pct(r.degraded_rate)}）`),
        statRow('没有推荐', num(r.no_suggestion)),
        statRow('建议转人工', num(r.transfers)),
        statRow('端到端 P95 ≤ 5 秒（§12.1）', p95),
      ]),
      countTable('硬规则命中（含自动修正）', '规则', r.hard_hit_counts, (k) => label(L.hardRules, k)),
      countTable('降级', '环节', r.degraded_counts, degradedLabel),
    );
  }

  // ================================================================ 启动

  const LOADERS = {
    knowledge: loadKnowledge,
    policies: loadPolicies,
    items: loadItemsTab,
    config: loadConfig,
    rollout: loadRollout,
    review: loadReviewTab,
    reports: loadReports,
  };

  function init() {
    if (CONFIG.platform_name) $('platform-name').textContent = CONFIG.platform_name;
    fillOptions($('faq-status'), L.knowledgeStatus);
    const approved = $('faq-status').querySelector('option[value="approved"]');
    if (approved) approved.defaultSelected = true; // FAQ 列表默认只看已上线的（候选知识在上面单独列出）
    fillOptions($('faq-intent'), L.intents);
    fillOptions($('review-filter'), L.reviewStatus);
    if (LIMITS.question) $('faq-question').maxLength = LIMITS.question;
    if (LIMITS.answer) $('faq-answer').maxLength = LIMITS.answer;
    if (LIMITS.sample_size_max) $('review-size').max = String(LIMITS.sample_size_max);
    $('review-size').defaultValue = String(LIMITS.sample_size || 50); // 表单 reset 后回到默认条数

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
    for (const button of tabButtons()) button.addEventListener('click', () => selectTab(button.dataset.tab));
    $('tabs').addEventListener('keydown', onTabKey);
    for (const button of document.querySelectorAll('[data-refresh]')) {
      button.addEventListener('click', () => {
        if (state.me) LOADERS[button.dataset.refresh]();
      });
    }
    $('faq-filter').addEventListener('submit', (event) => event.preventDefault());
    $('faq-status').addEventListener('change', () => {
      if (state.me) loadKnowledge();
    });
    $('faq-form').addEventListener('submit', createFaq);
    $('sync-all').addEventListener('click', () => {
      if (isMerchant()) syncItems(null, []);
    });
    $('review-form').addEventListener('submit', (event) => {
      event.preventDefault();
      if (!state.me) return;
      const p = periodValue();
      if (p.error) {
        setStatus($('review-status'), p.error, 'error');
        $('review-period').focus();
        return;
      }
      setStatus($('review-status'), '', '');
      loadQualityReport(p.value);
    });
    $('review-sample').addEventListener('click', sample);
    $('review-list-form').addEventListener('submit', (event) => {
      event.preventDefault();
      loadQueue();
    });
    $('report-form').addEventListener('submit', (event) => {
      event.preventDefault();
      if (state.me) loadReports();
    });

    const saved = storage.get(STORAGE_TOKEN);
    if (saved) signIn(saved);
    else $('token').focus();
  }

  init();
})();
