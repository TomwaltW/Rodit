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
 * - 两个页面共用 sessionStorage 里的令牌：退出、令牌无效时只在存着的还是本页的令牌时才清（forgetToken），
 *   不清掉另一个页面后来登录的人；从往返缓存（bfcache）恢复时存着的令牌变了（另一个页面退出或换人登录），
 *   先丢掉本页内存里的身份（不动 sessionStorage），有令牌再用它登录；
 * - 同一块区域的加载按请求编号只用最近一次的结果（nextRun / latest）：换筛选、换商品、换周期后，
 *   晚回来的旧响应不覆盖新的；防重复提交的 busy 记下发起的身份，退出后不影响下一位的按钮；
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
    stored: null, // 本页写进 sessionStorage 的令牌（退出时只清自己的；存不了时为 null）
    me: null,
    clockSkew: 0, // 服务端时钟减本机时钟（毫秒），登录时按 /v1/me 的 server_time 算
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

  /** 把登录用的令牌存进 sessionStorage，记下是本页存的（存不了时不记：两个页面本来也共用不了）。 */
  function keepToken(token) {
    storage.set(STORAGE_TOKEN, token);
    state.stored = storage.get(STORAGE_TOKEN) === token ? token : null;
  }

  /**
   * 清掉 sessionStorage 里的令牌、话术助手的会话和页签偏好——只在存着的还是这个令牌时才清：
   * 另一个页面后来登录的人（同一标签页里后退、前进回来时）不受本页退出或 401 的影响。
   */
  function forgetToken(token) {
    if (!token || storage.get(STORAGE_TOKEN) !== token) return;
    storage.remove(STORAGE_TOKEN);
    storage.remove(STORAGE_SESSION);
    storage.remove(STORAGE_TAB);
  }

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

  /**
   * 防重复提交：同一个 ctx 上一次还没完成时直接返回；执行期间禁用 controls。
   * ctx 记下发起的身份和第几次：退出或换人登录后，上一位没完成的那次不挡下一位，
   * 它晚回来时也不再改按钮和标记（只有最近一次能收尾）。
   */
  async function busy(ctx, controls, task) {
    if (ctx.busy && ctx.owner === state.me) return undefined;
    const run = (ctx.run = (ctx.run || 0) + 1);
    ctx.busy = true;
    ctx.owner = state.me;
    const list = controls.filter(Boolean);
    for (const control of list) control.disabled = true;
    try {
      return await task();
    } finally {
      if (ctx.run === run) {
        ctx.busy = false;
        ctx.owner = null;
        for (const control of list) control.disabled = false;
      }
    }
  }

  // 页面上固定的按钮（不随列表重建）共用的防重复提交标记：退出时复位，按钮重新可用
  const sharedBusy = [];

  function sharedCtx(...controlIds) {
    const ctx = { busy: false, controlIds };
    sharedBusy.push(ctx);
    return ctx;
  }

  function resetBusy() {
    for (const ctx of sharedBusy) {
      ctx.run = (ctx.run || 0) + 1; // 上一位没完成的那次晚回来时不再收尾
      ctx.busy = false;
      ctx.owner = null;
      for (const id of ctx.controlIds) $(id).disabled = false;
    }
  }

  // 请求顺序：同一块区域只用最近一次请求的结果（换筛选、换商品、换周期、收起之后，晚回来的旧响应丢掉）
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

  /** 表单里各输入框的当前值（判断有没有没保存的改动）。 */
  function formValues(root) {
    return JSON.stringify(
      Array.from(root.querySelectorAll('input, select, textarea'), (c) =>
        c.type === 'checkbox' || c.type === 'radio' ? c.checked : c.value,
      ),
    );
  }

  /** 区域里有打开的面板（修改、确认），或者焦点在里面：原地刷新会丢掉正在写的内容。 */
  function busyEditing(selector) {
    for (const root of document.querySelectorAll(selector)) {
      if (root.querySelector('.entry-panel > *') || root.contains(document.activeElement)) return true;
    }
    return false;
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

  let signIns = 0; // 第几次登录：退出、后退恢复时加一，还没回来的登录作废

  async function signIn(token) {
    if (state.signingIn) return false;
    const run = (signIns += 1);
    state.signingIn = true;
    clearError();
    state.token = token;
    const button = $('login-button');
    button.disabled = true;
    let me;
    try {
      me = await api('GET', 'v1/me');
    } catch (err) {
      if (run !== signIns) return false; // 期间退出或后退恢复时换了令牌：这次登录作废
      state.token = null;
      // 令牌无效才丢掉；网络、服务故障时留着，刷新后还能直接登录
      if (err.status === 401 || err.status === 403) forgetToken(token);
      if (err.code === 'forbidden_role') {
        showError('这个令牌是平台对接用的，不能登录网页工具；请用店长、商家或复核人员的令牌。');
      } else {
        showError(err.message);
      }
      $('token').focus();
      return false;
    } finally {
      if (run === signIns) {
        state.signingIn = false;
        button.disabled = false;
      }
    }
    if (run !== signIns) return false;
    if (!me || !inGroup('person', me.role)) {
      state.token = null;
      forgetToken(token);
      showError('这个令牌不能登录网页工具。');
      $('token').focus();
      return false;
    }
    if (!tabsFor(me.role).length) {
      // 客服：控制台没有可用的功能。令牌留着，同一标签页打开话术助手时直接登录；
      // 页头显示是谁、给「退出」按钮，不想交给话术助手时可以丢掉
      state.token = null;
      keepToken(token);
      $('token').value = '';
      showIdentity(me);
      showError(`${label(L.roles, me.role)}请使用话术助手，控制台只给店长、商家和复核人员用。`, {
        href: 'assist',
        text: '打开话术助手',
      });
      return false;
    }
    state.me = me;
    state.clockSkew = clockSkewOf(me);
    keepToken(token);
    enterApp();
    return true;
  }

  function showIdentity(me) {
    $('whoami').hidden = false;
    $('identity').textContent = `${label(L.roles, me.role)} ${me.user_id} · 店铺 ${me.shop_id}`;
    // 售后审核工作台只给客服、店长和商家（和售后审核接口的角色一致）：复核人员不显示这个链接
    $('aftersale-link').hidden = !inGroup('human', me.role);
  }

  function enterApp() {
    const me = state.me;
    $('token').value = '';
    $('login').hidden = true;
    $('app').hidden = false;
    showIdentity(me);
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

  /**
   * 退出：清掉本页的身份、数据和表单。sessionStorage 只在存着的还是本页的令牌时才清（forgetToken）；
   * opts.keepStorage：只清本页（后退恢复时发现另一个页面已经退出或换人登录，存着的令牌归那边）。
   */
  function signOut(opts = {}) {
    if (!opts.keepStorage) forgetToken(state.stored);
    signIns += 1; // 还没回来的登录作废
    state.signingIn = false;
    $('login-button').disabled = false;
    state.token = null;
    state.stored = null;
    state.me = null;
    state.tab = null;
    state.loaded = {};
    state.items = [];
    state.policies = [];
    state.config = null;
    state.stageMessage = null;
    state.review = { period: null, items: [] };
    state.itemOpener = null;
    dropRuns();
    resetBusy();
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
    $('aftersale-link').hidden = false; // 没登录时谁都可能用得上：照常显示
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

  /**
   * 写操作影响了别的页签的数据（比如补了政策，「要补的知识」会变）：别的页签下次切过去时重新加载；
   * 写操作还没回来时已经切过去的（正显示的页签）马上原地刷新，不关掉正在编辑的表单（REFRESHERS）。
   */
  function stale(...ids) {
    for (const id of ids) {
      if (id !== state.tab) state.loaded[id] = false;
      else if (state.loaded[id]) (REFRESHERS[id] || LOADERS[id])();
    }
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

  // 商品列表是知识页、商品页共用的缓存：比已经用上的那次请求新才用（晚回来的旧列表不覆盖新的）
  const itemRuns = { sent: 0, used: 0 };

  async function fetchItems() {
    const me = state.me;
    const run = (itemRuns.sent += 1);
    const list = await api('GET', shopPath('console/items'));
    if (state.me === me && run > itemRuns.used) {
      itemRuns.used = run;
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

  const faqCreateCtx = sharedCtx('faq-submit');

  async function loadKnowledge() {
    const me = state.me;
    const gapsRun = nextRun('gaps');
    const listRun = nextRun('faqs');
    for (const id of ['gaps', 'candidates', 'faqs']) loading($(id));
    const [gaps, candidates, faqs] = await Promise.allSettled([
      api('GET', shopPath('knowledge/gaps')),
      api('GET', shopPath('console/faqs', { status: 'candidate' })),
      api('GET', shopPath('console/faqs', { status: $('faq-status').value })),
      fetchItems().catch(() => null), // 下拉框和商品名用；失败时显示商品 ID
    ]);
    if (state.me !== me) return; // 期间退出或换人登录：不把上一位店铺的数据显示给下一位
    if (latest('gaps', gapsRun)) {
      if (gaps.status === 'fulfilled') renderGaps(gaps.value);
      else loadFailed($('gaps'), '要补的知识', gaps.reason);
    }
    if (!latest('faqs', listRun)) return; // 期间又换了筛选条件：以后一次为准
    if (candidates.status === 'fulfilled') renderFaqList($('candidates'), candidates.value, '没有待审核的候选知识。');
    else loadFailed($('candidates'), '候选知识', candidates.reason);
    if (faqs.status === 'fulfilled') renderFaqList($('faqs'), faqs.value, '没有符合条件的 FAQ。');
    else loadFailed($('faqs'), 'FAQ 列表', faqs.reason);
  }

  /** 只刷新「要补的知识」（知识页有打开的修改面板时，别的列表不动）。 */
  async function loadGaps() {
    const me = state.me;
    const run = nextRun('gaps');
    loading($('gaps'));
    try {
      const gaps = await api('GET', shopPath('knowledge/gaps'));
      if (state.me !== me || !latest('gaps', run)) return;
      renderGaps(gaps);
    } catch (err) {
      if (state.me !== me || !latest('gaps', run)) return;
      loadFailed($('gaps'), '要补的知识', err);
    }
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
    const run = nextRun('policies');
    loading($('policies'));
    put($('policy-summary'));
    try {
      const rows = await api('GET', shopPath('console/policies'));
      if (state.me !== me || !latest('policies', run)) return;
      state.policies = Array.isArray(rows) ? rows : [];
      renderPolicySummary();
      put($('policies'), h('ul', { className: 'policy-list' }, state.policies.map((row) => policyRow(row))));
    } catch (err) {
      if (state.me !== me || !latest('policies', run)) return;
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
      if (filled) {
        const clear = h('button', { type: 'button', className: 'secondary', 'aria-describedby': titleId }, '清空');
        clear.addEventListener('click', () => openPolicyClear(ctx, clear));
        ctx.actions.append(clear);
      }
    }
    if (message) setStatus(ctx.status, message, 'ok');
    return ctx.node;
  }

  // 政策不再适用（如赠品活动结束）：清空后回答不再引用，"要补的知识"里会提醒
  function openPolicyClear(ctx, opener) {
    const label = ctx.row.label || ctx.row.key;
    const confirm = h('button', { type: 'button', className: 'danger' }, '确认清空');
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const box = h('div', { className: 'confirm' }, [
      h('p', null, `清空后回答不再引用「${label}」；以后需要时再填写。`),
      h('div', { className: 'actions tight' }, [confirm, cancel]),
    ]);
    const close = () => {
      ctx.panel.replaceChildren();
      ctx.actions.hidden = false;
      opener.focus();
    };
    cancel.addEventListener('click', close);
    confirm.addEventListener('click', async () => {
      const me = state.me;
      clearError();
      await busy(ctx, [confirm, cancel], async () => {
        try {
          const row = await api('POST', shopPath(`console/policies/${enc(ctx.row.key)}/clear`), {});
          if (state.me !== me) return;
          state.policies = state.policies.map((r) => (r.key === row.key ? row : r));
          renderPolicySummary();
          stale('knowledge');
          const message = `已清空「${row.label}」。`;
          const fresh = policyRow(row, message);
          ctx.node.replaceWith(fresh);
          fresh.querySelector('.status').focus();
          announce(message);
        } catch (err) {
          if (state.me !== me) return;
          reportFailure(ctx.status, '没有清空成功：', err);
        }
      });
    });
    ctx.actions.hidden = true;
    ctx.panel.replaceChildren(box);
    confirm.focus();
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

  const syncCtx = sharedCtx('sync-all');

  async function loadItemsTab() {
    const me = state.me;
    const run = nextRun('itemTable');
    loading($('items'));
    try {
      await fetchItems();
      if (state.me !== me || !latest('itemTable', run)) return;
      renderItemTable();
    } catch (err) {
      if (state.me !== me || !latest('itemTable', run)) return;
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

  /** 打开商品卡片。drafts：同步前商家改过、还没保存的字段（{字段: 输入框里的文字}），重新显示后放回去。 */
  async function loadItemDetail(itemId, opener, focus = true, drafts = null) {
    const me = state.me;
    const run = nextRun('item');
    const box = $('item-detail');
    state.itemOpener = opener || null;
    box.hidden = false;
    box.dataset.itemId = itemId;
    loading(box);
    try {
      const card = await api('GET', shopPath(`console/items/${enc(itemId)}`));
      if (state.me !== me || !latest('item', run)) return; // 期间点了别的商品或收起：以后一次为准
      renderItemDetail(card, null, drafts);
      if (focus) box.focus();
    } catch (err) {
      if (state.me !== me || !latest('item', run)) return;
      loadFailed(box, `商品 ${itemId}`, err);
    }
  }

  /** 打开的商品卡片里商家改过、还没保存的字段（和显示时的内容比）。 */
  function itemDrafts() {
    const box = $('item-detail');
    const drafts = {};
    if (box.hidden) return drafts;
    for (const area of box.querySelectorAll('.item-form textarea[data-field]')) {
      if (area.value !== area.defaultValue) drafts[area.dataset.field] = area.value;
    }
    return drafts;
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

  function renderItemDetail(card, message, drafts) {
    const box = $('item-detail');
    const titleId = uid('item-title');
    const close = h('button', { type: 'button', className: 'secondary' }, '收起');
    close.addEventListener('click', () => {
      nextRun('item'); // 还没回来的商品卡片不再显示
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
    const merchant = isMerchant() ? itemForm(card, message, drafts) : itemReadonly(card);
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

  function itemForm(card, message, drafts) {
    const areas = {};
    const kept = []; // 同步后放回去的、还没保存的字段
    const nodes = Object.keys(L.itemFields).map((f) => {
      const id = uid(`item-${f}`);
      const lim = ITEM_LIMITS[f] || {};
      const isList = f !== 'size_advice';
      const area = h('textarea', { id, rows: isList ? '3' : '2', 'data-field': f });
      if (!isList && lim.max_chars) area.maxLength = lim.max_chars;
      // defaultValue 是服务端的内容：和 value 不同就是商家改过、还没保存
      area.defaultValue = isList ? (Array.isArray(card[f]) ? card[f].join('\n') : '') : card[f] || '';
      if (drafts && has(drafts, f) && drafts[f] !== area.defaultValue) {
        area.value = drafts[f];
        kept.push(label(L.itemFields, f));
      }
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
    if (kept.length) {
      const note = `平台字段已按同步结果更新；你改过的「${kept.join('」「')}」还没保存，已保留在输入框里，确认后点「保存商家填写的字段」。`;
      setStatus(status, note, 'warn');
      announce(note);
    } else if (message) {
      setStatus(status, message, 'ok');
    }
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
    const run = nextRun('itemTable');
    try {
      await fetchItems();
      if (state.me !== me || !latest('itemTable', run)) return;
      renderItemTable();
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
        if (shown && state.me === me) {
          // 平台字段可能变了：重新显示卡片，商家改过、还没保存的字段放回输入框（不替商家保存，也不丢）
          await loadItemDetail(shown, state.itemOpener, false, itemDrafts());
        }
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
    const run = nextRun('config');
    loading($('config'));
    try {
      const view = await api('GET', shopPath('console/config'));
      if (state.me !== me || !latest('config', run)) return;
      state.config = view;
      put($('config'), h('div', { className: 'config-list' }, CONFIG_FIELDS.map((f) => configRow(f))));
    } catch (err) {
      if (state.me !== me || !latest('config', run)) return;
      loadFailed($('config'), '店铺设置', err);
    }
  }

  const pristine = new WeakMap(); // 设置行的编辑表单 → 刚显示时的输入值

  /** 这一行有没保存的改动，或者焦点在里面：原地刷新时留着不换。 */
  function rowEdited(row) {
    if (row.contains(document.activeElement)) return true;
    const form = row.querySelector('form');
    return Boolean(form && pristine.has(form) && pristine.get(form) !== formValues(form));
  }

  /** 设置页正显示时原地刷新（比如切换阶段后影子模式的开关变了）：只换掉没在编辑的行。 */
  async function refreshConfig() {
    const me = state.me;
    const run = nextRun('config');
    try {
      const view = await api('GET', shopPath('console/config'));
      if (state.me !== me || !latest('config', run)) return;
      state.config = view;
      const list = $('config').querySelector('.config-list');
      if (!list) {
        put($('config'), h('div', { className: 'config-list' }, CONFIG_FIELDS.map((f) => configRow(f))));
        return;
      }
      for (const fieldDef of CONFIG_FIELDS) {
        const old = list.querySelector(`[data-key="${fieldDef.key}"]`);
        if (old && rowEdited(old)) continue; // 正在改的行留着：保存时只提交这一项
        const fresh = configRow(fieldDef);
        if (old) old.replaceWith(fresh);
        else list.append(fresh);
      }
    } catch (err) {
      if (state.me !== me || !latest('config', run)) return;
      if ($('config').querySelector('.config-list')) showError(`店铺设置没有刷新：${err.message}`);
      else loadFailed($('config'), '店铺设置', err);
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
    pristine.set(form, formValues(form));
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

  /** 上线状态。keepControls：切换阶段的面板打开着时只刷新上面的状态（抽样、打分后原地刷新）。 */
  async function loadRollout(keepControls = false) {
    const me = state.me;
    const run = nextRun('rollout');
    const controlsRun = keepControls ? null : nextRun('stage');
    loading($('rollout'));
    if (!keepControls) put($('stage-controls'));
    try {
      const status = await api('GET', shopPath('rollout/status'));
      if (state.me !== me) return;
      if (latest('rollout', run)) renderRollout(status);
      if (!keepControls && latest('stage', controlsRun)) renderStageControls(status);
    } catch (err) {
      if (state.me !== me || !latest('rollout', run)) return;
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
            h('p', null, '需要人工确认的条件（缺数据，切换时由复核人员确认）：'),
            h(
              'ul',
              null,
              unchecked.map((u) => h('li', null, u.reason ? `${u.description || u.name}：${u.reason}` : u.description || u.name)),
            ),
          ])
        : null,
      currentGates(s).map((g) => (g.gate ? gateTable(g.gate, g.title) : null)),
      evidenceSection(s.evidence),
      autoScheduleSection(s.auto_schedule),
      semiAutoSection(s.semi_auto),
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

  /** 登记的周 → 条数；必须登记而没登记的周写"未登记"（§13.4，复核人员按周登记）。 */
  function incidentText(t) {
    if (!t) return '—';
    const byWeek = t.by_week || {};
    const weeks = Array.isArray(t.weeks) ? t.weeks : [];
    const extra = Object.keys(byWeek).filter((w) => !weeks.includes(w));
    const parts = [...weeks, ...extra].map((w) => `${w} ${has(byWeek, w) ? num(byWeek[w]) : '未登记'}`);
    const total = t.count === null || t.count === undefined ? '缺登记' : `合计 ${num(t.count)}`;
    return `${total}（${parts.join('、') || '—'}）`;
  }

  /** 合规扫描汇总：计数和推荐 / 消息编号，没有文字。 */
  function scanText(c) {
    if (!c) return '—';
    const ids = [
      ...(Array.isArray(c.suggestion_ids) ? c.suggestion_ids.map((id) => `推荐 #${id}`) : []),
      ...(Array.isArray(c.message_ids) ? c.message_ids.map((id) => `消息 ${id}`) : []),
    ];
    const listed = ids.length ? `：${ids.join('、')}${c.ids_truncated ? ' …' : ''}` : '';
    return (
      `${fmtRange(c.since, c.until)} 编辑后发送的推荐 ${num(c.scanned_suggestions)} 条、发出的消息 ${num(c.scanned_messages)} 条，` +
      `疑似违规 ${num(c.flagged)} 条${listed}`
    );
  }

  /** 门槛里原来没有数据来源的条件现在的依据：测试集结论、平台违规 / 投诉登记、合规扫描。 */
  function evidenceSection(ev) {
    if (!ev) return null;
    const t = ev.test_set || {};
    const testSet = t.passed === null || t.passed === undefined ? t.reason || '没有能用的评测结论' : t.detail || (t.passed ? '达标' : '没达标');
    return [
      h('h3', null, '门槛依据'),
      h('dl', { className: 'stats' }, [
        statRow('测试集（评测结论）', testSet),
        statRow('平台违规（复核人员登记）', incidentText(ev.platform_violations)),
        statRow('投诉（复核人员登记）', incidentText(ev.complaints)),
        statRow('合规扫描（代理指标）', scanText(ev.compliance)),
      ]),
    ];
  }

  async function loadShadowDetail(box) {
    const me = state.me;
    const run = nextRun('shadow');
    loading(box);
    try {
      const view = await api('GET', shopPath('rollout/shadow'));
      if (state.me !== me || !latest('shadow', run)) return;
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
      if (state.me !== me || !latest('shadow', run)) return;
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

  // ---------------------------------------------------------------- 全自动生效时段（§15 Phase 3）、半自动（§5.6、§13.4）

  const SCHEDULE_TEXT = { off_hours: '只在客服工作时间以外', always: '全天' };
  const AUTO_GATE_TEXT = {
    service_hours: '现在是客服工作时间，机器人接待的会话转给在岗客服',
    no_service_hours: '店铺没填客服工作时间，不知道客服在不在，先不自动回复',
    unknown_time: '不知道现在的时间，先不自动回复',
  };
  const SEMI_BLOCKED_TEXT = { high_risk: '高风险类型，不能开', not_answer: '不走推荐回答，不能开' };
  const semiAuto = { message: null }; // 开关成功后，刷新上线状态时显示的结果

  /** 全自动什么时候生效：只读（总开关和生效时段由我们评估后调整，工作时间由商家在设置页填）。 */
  function autoScheduleSection(a) {
    if (!a) return null;
    const hours = Array.isArray(a.service_hours) ? a.service_hours : [];
    let now = '没填客服工作时间';
    if (a.in_service_hours === true) now = '客服工作时间内';
    else if (a.in_service_hours === false) now = '客服工作时间外';
    let auto = '没开（全自动要平台准入，由我们评估后开启）';
    if (a.auto_reply_enabled) {
      auto = a.auto_active_now
        ? '生效：机器人接待的高置信度、低风险问题会自动回复'
        : `暂停：${label(AUTO_GATE_TEXT, a.auto_gate)}`;
    }
    return [
      h('h3', null, '全自动生效时段（§15 Phase 3：先在非工作时间试行）'),
      h('dl', { className: 'stats', id: 'auto-schedule' }, [
        statRow('全自动总开关', a.auto_reply_enabled ? '开' : '关'),
        statRow('生效时段', label(SCHEDULE_TEXT, a.auto_reply_schedule)),
        statRow(`客服工作时间（${a.timezone || '北京时间'}）`, hours.length ? hours.join('、') : '没填'),
        statRow('现在', now),
        statRow('全自动现在', auto),
        statRow('下一次进出工作时间', a.next_switch_at ? fmtTime(a.next_switch_at) : '—'),
      ]),
      h('p', { className: 'hint' }, '客服工作时间由商家在「设置」页填写；全自动的开关和生效时段由我们评估后调整，这里只能查看。'),
    ];
  }

  /** 门槛里每一周的原样采纳率和推荐组数（feedback.rollout.semi_auto_gate 的 period_N 条件）。 */
  function gateWeeks(gate) {
    const checks = gate && Array.isArray(gate.checks) ? gate.checks : [];
    const weeks = checks.filter((c) => /^period_\d+\./.test(String(c.name)));
    if (!weeks.length) return '还没有数据';
    return weeks.map((c, i) => `第 ${i + 1} 周 ${pct(c.value)}（${c.detail || '—'}）`).join('；');
  }

  /** 半自动（按问题类型）：开了的类型、门槛；复核人员能开（门槛没过要强制并写原因），商家和复核人员能关。 */
  function semiAutoSection(sa) {
    if (!sa) return null;
    const role = state.me ? state.me.role : null;
    const canEnable = inGroup('reviewer', role);
    const canDisable = inGroup('stage_shadow', role); // 关闭（安全方向）：商家、复核人员
    const status = h('p', { className: 'status', role: 'status', tabindex: '-1', id: 'semi-auto-status' });
    const panel = h('div', { className: 'entry-panel' });
    const ui = { ctx: { busy: false }, status, panel, buttons: [] };
    const intents = Array.isArray(sa.intents) ? sa.intents : [];
    const shown = intents.filter((r) => r.allowed || r.enabled);
    const blocked = intents.filter((r) => !r.allowed && !r.enabled);
    const rows = shown.map((r) => semiAutoRow(r, ui, canEnable, canDisable));
    const enabled = Array.isArray(sa.enabled) ? sa.enabled : [];
    const nodes = [
      h('h3', null, '半自动（§5.6：高置信度的推荐直接填进客服输入框，客服按发送即可）'),
      h('p', { className: 'hint' }, [
        `开启条件（§13.4）：该类型最近 ${num(sa.periods)} 个完整周每周至少 ${num(sa.min_groups)} 个推荐组、`,
        `原样采纳率都 ≥ ${pct(sa.min_as_is)}；开了以后问题类型置信度 ≥ ${pct(sa.threshold)}、`,
        '没有提醒和降级的推荐才填进输入框，仍由客服看过再发。',
      ]),
      h('p', { id: 'semi-auto-enabled' }, `已开半自动：${enabled.length ? enabled.map((i) => label(L.intents, i)).join('、') : '没有'}`),
      table('各问题类型的半自动', ['问题类型', '状态', '门槛（原样采纳率）', '操作'], rows, true),
      blocked.length
        ? h('p', { className: 'hint' }, `不能开半自动：${blocked.map((r) => `${label(L.intents, r.intent)}（${label(SEMI_BLOCKED_TEXT, r.blocked_reason)}）`).join('、')}`)
        : null,
      !canEnable && !canDisable ? h('p', { className: 'muted' }, '半自动由复核人员按门槛开启，商家可以关闭。') : null,
      !canEnable && canDisable ? h('p', { className: 'hint' }, '商家可以关闭某个类型的半自动；开启由复核人员确认门槛后操作。') : null,
      panel,
      status,
    ];
    if (semiAuto.message) {
      setStatus(status, semiAuto.message, 'ok');
      semiAuto.message = null;
      window.setTimeout(() => status.focus(), 0);
    }
    return h('div', { className: 'semi-auto', id: 'semi-auto' }, nodes);
  }

  function semiAutoRow(r, ui, canEnable, canDisable) {
    const name = label(L.intents, r.intent);
    let current;
    if (r.enabled && !r.allowed) current = h('span', { className: 'badge bad' }, '已开（不该开，请关闭）');
    else current = h('span', { className: r.enabled ? 'badge ok' : 'badge' }, r.enabled ? '已开' : '没开');
    let gate = label(SEMI_BLOCKED_TEXT, r.blocked_reason);
    if (r.gate) {
      gate = [
        h('span', { className: r.gate.passed ? 'badge ok' : 'badge bad' }, r.gate.passed ? '达到' : '没达到'),
        ' ',
        gateWeeks(r.gate),
      ];
    }
    let action = '—';
    if (r.enabled && canDisable) {
      const button = h('button', { type: 'button', className: 'secondary', 'aria-label': `关闭「${name}」的半自动` }, '关闭');
      button.addEventListener('click', () => postSemiAuto({ intent: r.intent, enabled: false }, { ...ui, name, controls: ui.buttons }));
      ui.buttons.push(button);
      action = button;
    } else if (!r.enabled && r.allowed && canEnable) {
      const button = h('button', { type: 'button', 'aria-label': `开启「${name}」的半自动` }, '开启');
      button.addEventListener('click', () => postSemiAuto({ intent: r.intent, enabled: true }, { ...ui, name, controls: ui.buttons }));
      ui.buttons.push(button);
      action = button;
    }
    return h('tr', { 'data-intent': r.intent }, [
      h('th', { scope: 'row' }, name),
      h('td', null, current),
      h('td', { className: 'wrap' }, gate),
      h('td', null, action),
    ]);
  }

  async function postSemiAuto(body, ui) {
    const me = state.me;
    clearError();
    await busy(ui.ctx, ui.controls, async () => {
      try {
        const change = await api('POST', shopPath('rollout/semi-auto'), body);
        if (state.me !== me) return;
        const what = change.enabled ? '开启' : '关闭';
        semiAuto.message = change.changed
          ? `已${what}「${ui.name}」的半自动${change.forced ? '（门槛没达到，强制开启，原因已记入审计）' : ''}。`
          : `「${ui.name}」的半自动本来就是${change.enabled ? '开' : '关'}着的，没有改动。`;
        announce(semiAuto.message);
        await loadRollout(true);
      } catch (err) {
        if (state.me !== me) return;
        if (err.status === 409 && err.code === 'gate_not_passed' && err.data && err.data.gate) {
          showSemiAutoForce(err.data, body, ui);
          return;
        }
        reportFailure(ui.status, '没有改：', err);
      }
    });
  }

  /** 开半自动的门槛没过（409）：显示门槛结果，复核人员可以写明原因后强制开启。 */
  function showSemiAutoForce(data, body, ui) {
    const titleId = uid('semi-fail');
    const noteId = uid('semi-note');
    const note = h('textarea', { id: noteId, rows: '2', maxlength: String(LIMITS.note) });
    const force = h('button', { type: 'submit', className: 'danger' }, '强制开启');
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const close = () => {
      ui.panel.replaceChildren();
      const row = $('semi-auto') && $('semi-auto').querySelector(`tr[data-intent="${body.intent}"] button`);
      if (row) row.focus();
    };
    cancel.addEventListener('click', close);
    const form = h('form', { className: 'entry-form', novalidate: true }, [
      field(noteId, `强制开启「${ui.name}」半自动的原因（必填，记入审计）`, note, '比如：原样采纳率接近门槛、已人工抽查过这类推荐。'),
      h('div', { className: 'actions tight' }, [force, cancel]),
    ]);
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = note.value.trim();
      if (!text) {
        setStatus(ui.status, '强制开启要写明原因。', 'error');
        note.focus();
        return;
      }
      postSemiAuto({ ...body, force: true, note: text }, { ...ui, controls: [...ui.buttons, force, cancel] });
    });
    form.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        close();
      }
    });
    const box = h('div', { className: 'gate-failure', role: 'group', 'aria-labelledby': titleId }, [
      h('p', { id: titleId, className: 'warnings-title', tabindex: '-1' }, `门槛没过，没有开启：${data.message || ''}`),
      gateTable(data.gate, `半自动 · ${ui.name}`),
      form,
    ]);
    ui.panel.replaceChildren(box);
    setStatus(ui.status, '', '');
    $(titleId).focus();
    announce('门槛没过，没有开启半自动');
  }

  // ================================================================ 每周复核（§13.5，按 §13.2 打分）

  const sampleCtx = sharedCtx('review-sample', 'review-report-button');
  const queueCtx = sharedCtx('review-load');

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
    const run = nextRun('qualityReport');
    const box = $('review-report');
    loading(box);
    try {
      const report = await api('GET', shopPath('quality/report', { period }));
      if (state.me !== me || !latest('qualityReport', run)) return; // 期间又查了别的周期：以后一次为准
      if (fillPeriod && report.period && !$('review-period').value) $('review-period').value = report.period;
      renderQualityReport(report);
    } catch (err) {
      if (state.me !== me || !latest('qualityReport', run)) return;
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
      const run = nextRun('queue');
      loading(box);
      try {
        const list = await api('GET', shopPath('quality/reviews', { period: p.value, status: $('review-filter').value }));
        if (state.me !== me || !latest('queue', run)) return;
        state.review = { period: list.period, items: Array.isArray(list.items) ? list.items : [] };
        renderQueue(list);
        if (!periodOverride) await loadQualityReport(list.period); // 报表和列表看同一个周期
      } catch (err) {
        if (state.me !== me || !latest('queue', run)) return;
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

  // 服务端时钟（/v1/me 的 server_time）减本机时钟：统计范围按服务端的时间算，
  // 本机时钟不准、或者本地演示的时钟固定在合成数据的日期时，"近 7 天"照样对得上
  function clockSkewOf(me) {
    const t = Date.parse(me && me.server_time);
    return Number.isFinite(t) ? t - Date.now() : 0;
  }

  function sinceParam() {
    const days = Number($('range').value);
    if (!days) return null;
    const now = Date.now() + (state.clockSkew || 0);
    return new Date(now - days * 86400000).toISOString();
  }

  async function loadReports() {
    const me = state.me;
    const run = nextRun('reports');
    loading($('adoption'));
    loading($('qa-report'));
    const since = sinceParam();
    const [adoption, qa] = await Promise.allSettled([
      api('GET', shopPath('reports/adoption', { since })),
      api('GET', shopPath('reports/qa', { since })),
    ]);
    if (state.me !== me || !latest('reports', run)) return; // 期间又换了统计范围：以后一次为准
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
    if (!o || (!o.total && !(report.auto && report.auto.sent))) {
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
        report.auto && report.auto.sent
          ? statRow('全自动发送（不计入采纳率）', `${num(report.auto.sent)} 条 / ${num(report.auto.groups)} 组消息`)
          : null,
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

  // ================================================================ 平台客服指标 · 接入前后对比（§12.4、§1.3）
  //
  // 报表页的一节（页面模板里没有，启动时由 initPlatformMetrics 建好）：接入日期 + 每段天数 → 每项指标
  // 接入前 / 接入后的均值、变化（↑↓，按指标的好坏方向标绿 / 红）、数据来源、口径说明，和按周的柱状图
  // （SVG，createElementNS 画，文字一律 textContent）；「同步平台数据」给店长、商家，「登记」只给商家。
  // 接入日期：输入框里是用户填的才作为 go_live 发给接口；没填时用报表给的（上线记录 / 开通授权），
  // 报表也不知道时先按 28 天前算并提示填写。

  const PM_SPECS = CONFIG.platform_metrics || {};
  const PM_DEFAULT_WINDOW = 28;
  const PM_WINDOWS = [14, 28, 56];
  const PM_UNIT_TEXT = { ratio: '%', seconds: '秒', hours: '小时' };
  const PM_SOURCES = { platform: '平台数据', merchant: '商家登记', computed: '系统估算' };
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const pmSyncCtx = sharedCtx('pm-sync');
  const pmRecordCtx = sharedCtx('pm-record-submit');

  /** 本平台考核的指标（页面配置里的定义，按服务端的顺序）。 */
  function pmMetrics() {
    return Object.entries(PM_SPECS).filter(([, spec]) => spec && spec.platform === CONFIG.platform);
  }

  /** 服务端的今天往前 days 天（东八区的 YYYY-MM-DD）。 */
  function pmDaysAgo(days) {
    return fmtDate(new Date(Date.now() + (state.clockSkew || 0) - days * 86400000));
  }

  /** 一项指标的数值：比例显示成百分数，时长带单位。 */
  function pmValue(unit, value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    if (unit === 'ratio') return pct(value);
    return `${num(value)} ${PM_UNIT_TEXT[unit] || ''}`.trim();
  }

  /** 变化量：比例按百分点，时长带单位；带正负号。 */
  function pmDelta(unit, delta) {
    if (typeof delta !== 'number' || !Number.isFinite(delta)) return '—';
    const sign = delta > 0 ? '+' : delta < 0 ? '−' : '';
    const size = Math.abs(delta);
    if (unit === 'ratio') return `${sign}${Math.round(size * 1000) / 10} 个百分点`;
    return `${sign}${num(size)} ${PM_UNIT_TEXT[unit] || ''}`.trim();
  }

  function pmSources(sources) {
    const parts = Object.entries(sources || {}).map(([s, days]) => `${label(PM_SOURCES, s)} ${days} 天`);
    return parts.length ? parts.join('、') : '没有数据';
  }

  function pmWindowCell(stats, unit) {
    return h('td', null, [
      h('span', { className: 'pm-value' }, pmValue(unit, stats && stats.mean)),
      h('span', { className: 'pm-sub' }, stats ? pmSources(stats.sources) : '—'),
    ]);
  }

  function pmChangeCell(m) {
    const cls = m.direction === 'improved' ? 'good' : m.direction === 'worse' ? 'bad' : '';
    const arrow = typeof m.delta === 'number' && m.delta !== 0 ? (m.delta > 0 ? '↑ ' : '↓ ') : '';
    const rel = m.relative_change;
    const relative =
      typeof rel === 'number' && Number.isFinite(rel)
        ? `（${rel > 0 ? '+' : rel < 0 ? '−' : ''}${pct(Math.abs(rel))}）`
        : '';
    return h('td', null, [
      m.direction === 'insufficient'
        ? h('span', { className: 'muted' }, '—')
        : h('span', { className: `pm-change ${cls}`.trim() }, `${arrow}${pmDelta(m.unit, m.delta)}${relative}`),
      h('span', { className: `pm-sub pm-dir ${cls}`.trim() }, m.direction_label || ''),
    ]);
  }

  // ---------------------------------------------------------------- 按周的柱状图（SVG）

  function svgEl(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    }
    for (const child of [].concat(children || [])) {
      if (child === null || child === undefined) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  /** 坐标轴上限取整到好读的数（1、1.2、1.5、2、2.5、3、4、5、6、8 乘以 10 的幂）；比例最多到 100%。 */
  function pmNiceMax(max, unit) {
    if (!(max > 0)) return unit === 'ratio' ? 1 : 1;
    const mag = 10 ** Math.floor(Math.log10(max));
    const step = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((f) => f * mag >= max * 1.0001) || 10;
    const nice = step * mag;
    return unit === 'ratio' ? Math.min(nice, 1) : nice;
  }

  /** 柱子：数据端 4px 圆角、基线端方角（高度不够时退成矩形）。 */
  function pmBarPath(x, y, w, base) {
    const r = Math.min(4, w / 2, base - y);
    if (r <= 0.5) return `M${x},${base}V${y}H${x + w}V${base}Z`;
    return (
      `M${x},${base}V${y + r}A${r},${r} 0 0 1 ${x + r},${y}` +
      `H${x + w - r}A${r},${r} 0 0 1 ${x + w},${y + r}V${base}Z`
    );
  }

  function pmWeekName(p) {
    return p.phase === 'before' ? `接入前第 ${-p.index} 周` : `接入后第 ${p.index + 1} 周`;
  }

  /** 一项指标按周的柱状图：接入前灰、接入后蓝，接入日一条竖线；悬停 / 读屏看每周的值。 */
  function pmChart(m) {
    const weeks = Array.isArray(m.weekly) ? m.weekly : [];
    if (!weeks.length) return null;
    const W = 320;
    const H = 132;
    const top = 18;
    const bottom = 24;
    const left = 4;
    const right = 4;
    const base = H - bottom;
    const slot = (W - left - right) / weeks.length;
    const bar = Math.min(24, Math.max(6, slot - 8));
    const values = weeks.map((p) => (typeof p.mean === 'number' ? p.mean : 0));
    const yMax = pmNiceMax(Math.max(...values, 0), m.unit);
    const yOf = (v) => base - (v / yMax) * (base - top);
    const before = weeks.filter((p) => p.phase === 'before').length;
    const summary =
      `${m.label}按周均值：接入前 ${pmValue(m.unit, m.before && m.before.mean)}，` +
      `接入后 ${pmValue(m.unit, m.after && m.after.mean)}`;
    const nodes = [
      svgEl('title', null, summary),
      svgEl('line', { class: 'pm-grid', x1: left, x2: W - right, y1: top, y2: top }),
      svgEl('text', { class: 'pm-tick', x: left, y: top - 5 }, pmValue(m.unit, yMax)),
    ];
    weeks.forEach((p, i) => {
      const x = left + i * slot + (slot - bar) / 2;
      const has = typeof p.mean === 'number' && Number.isFinite(p.mean);
      const text = `${pmWeekName(p)}（${p.start} – ${p.end}）：${has ? pmValue(m.unit, p.mean) : '没有数据'}，${p.days} 天有数据`;
      // 命中区域比柱子大（整个格子），悬停 / 读屏都能看到这一周的值
      nodes.push(
        svgEl('g', { class: `pm-week ${p.phase}` }, [
          svgEl('title', null, text),
          svgEl('rect', { class: 'pm-hit', x: left + i * slot, y: top, width: slot, height: base - top }),
          has ? svgEl('path', { class: `pm-bar ${p.phase}`, d: pmBarPath(x, yOf(p.mean), bar, base) }) : null,
        ]),
      );
    });
    // 接入日：接入前最后一周和接入后第一周之间
    if (before > 0 && before < weeks.length) {
      const x = left + before * slot;
      nodes.push(svgEl('line', { class: 'pm-golive', x1: x, x2: x, y1: top - 14, y2: base }));
      nodes.push(svgEl('text', { class: 'pm-tick', x: x + 3, y: top - 5 }, '接入'));
    }
    nodes.push(svgEl('line', { class: 'pm-axis', x1: left, x2: W - right, y1: base, y2: base }));
    if (before > 0) {
      nodes.push(svgEl('text', { class: 'pm-tick', x: left + (before * slot) / 2, y: H - 6, 'text-anchor': 'middle' }, '接入前'));
    }
    if (before < weeks.length) {
      const mid = left + before * slot + ((weeks.length - before) * slot) / 2;
      nodes.push(svgEl('text', { class: 'pm-tick', x: mid, y: H - 6, 'text-anchor': 'middle' }, '接入后'));
    }
    const chart = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': summary, focusable: 'false' }, nodes);
    return h('figure', { className: 'pm-chart' }, [
      chart,
      h('figcaption', { className: 'pm-legend' }, [
        h('span', { className: 'pm-key before', 'aria-hidden': 'true' }),
        '接入前（每周均值）',
        h('span', { className: 'pm-key after', 'aria-hidden': 'true' }),
        '接入后',
      ]),
    ]);
  }

  // ---------------------------------------------------------------- 报表

  /** 报表页的「平台客服指标」：接入日期默认用报表给的，不知道时先按 28 天前算（见上）。 */
  async function loadPlatformMetrics() {
    const me = state.me;
    const run = nextRun('pm');
    const box = $('platform-metrics');
    loading(box);
    const input = $('pm-go-live');
    input.max = pmDaysAgo(0);
    const typed = input.dataset.auto ? '' : input.value;
    const windowDays = $('pm-window').value;
    try {
      let report = await api('GET', shopPath('reports/platform-metrics', { go_live: typed, window_days: windowDays }));
      if (state.me !== me || !latest('pm', run)) return; // 期间换了日期或换人登录：以后一次为准
      let guessed = false;
      if (report && !report.go_live && !typed) {
        const fallback = pmDaysAgo(PM_DEFAULT_WINDOW);
        report = await api('GET', shopPath('reports/platform-metrics', { go_live: fallback, window_days: windowDays }));
        if (state.me !== me || !latest('pm', run)) return;
        guessed = true;
      }
      if (report && report.go_live && !typed) {
        input.value = report.go_live;
        input.dataset.auto = 'yes'; // 不是用户填的：下次刷新照样让服务端决定
      }
      renderPlatformMetrics(report, guessed);
    } catch (err) {
      if (state.me !== me || !latest('pm', run)) return;
      if (err.code === 'platform_metrics_unavailable') {
        put(box, h('p', { className: 'muted' }, '本单元没有开通平台客服指标的对比。'));
        return;
      }
      loadFailed(box, '平台客服指标', err);
    }
  }

  function renderPlatformMetrics(report, guessed) {
    const box = $('platform-metrics');
    if (!report) {
      put(box, h('p', { className: 'muted' }, '没有数据。'));
      return;
    }
    const metrics = Array.isArray(report.metrics) ? report.metrics : [];
    const good = report.verdict === 'improved';
    const bad = report.verdict === 'worse' || report.verdict === 'mixed';
    const goLive = guessed
      ? `接入日期未知（没有上线记录，授权也没有开通时间）：先按 ${report.go_live} 计算，请在上面填写实际的接入日期。`
      : `接入日期 ${report.go_live || '未知'}（${report.go_live_label || ''}）`;
    const windows =
      report.before_start && report.after_end
        ? `接入前 ${report.before_start} – ${report.before_end}，接入后 ${report.after_start} – ${report.after_end}（各 ${report.window_days} 天，接入当天算接入后）`
        : '';
    const rows = metrics.map((m) =>
      h('tr', null, [
        h('th', { scope: 'row' }, [
          h('span', { className: 'pm-value' }, m.label),
          h('span', { className: 'pm-sub' }, m.higher_is_better ? '越高越好' : '越低越好'),
        ]),
        pmWindowCell(m.before, m.unit),
        pmWindowCell(m.after, m.unit),
        pmChangeCell(m),
      ]),
    );
    const details = metrics.map((m) =>
      h('div', { className: 'pm-metric' }, [
        h('h4', null, m.label),
        pmChart(m),
        h('p', { className: 'hint' }, `口径：${m.note}`),
        Array.isArray(m.flag_notes) && m.flag_notes.length
          ? h('ul', { className: 'pm-flags' }, m.flag_notes.map((t) => h('li', null, t)))
          : null,
      ]),
    );
    put(
      box,
      h('p', { className: 'headline' }, [
        h('strong', { className: good ? 'good' : bad ? 'bad' : null }, report.verdict_label || report.verdict),
        report.reliable ? '' : metrics.length ? '（有可靠性提示，仅供参考）' : '',
      ]),
      h('p', { className: guessed ? 'notice pm-guess' : 'hint' }, goLive),
      windows ? h('p', { className: 'hint' }, windows) : null,
      metrics.length ? table('平台客服指标接入前后对比', ['指标', '接入前', '接入后', '变化'], rows, true) : null,
      Array.isArray(report.notes) && report.notes.length
        ? h('ul', { className: 'pm-notes' }, report.notes.map((t) => h('li', null, t)))
        : null,
      details,
    );
  }

  /** 报表页：采纳率、质检（统计范围）和平台客服指标（接入日期）各自加载。 */
  function loadReportsTab() {
    loadReports();
    loadPlatformMetrics();
  }

  // ---------------------------------------------------------------- 同步、登记

  async function syncPlatformMetrics() {
    const me = state.me;
    const status = $('pm-status');
    await busy(pmSyncCtx, [$('pm-sync')], async () => {
      setStatus(status, '正在从平台同步最近 56 天的数据…', 'muted');
      try {
        const r = await api('POST', shopPath('platform-metrics/sync'));
        if (state.me !== me) return;
        const skipped = Object.values(r.skipped || {}).reduce((a, b) => a + b, 0);
        const message =
          `同步完成：${r.since} – ${r.until}，写入 ${r.written} 条` + (skipped ? `，跳过 ${skipped} 条` : '');
        setStatus(status, message, 'ok');
        announce(message);
        loadPlatformMetrics();
      } catch (err) {
        if (state.me !== me) return;
        if (err.code === 'metrics_source_not_configured') {
          setStatus(status, `${err.message}（下面「登记」）`, 'warn');
          return;
        }
        reportFailure(status, '同步没有完成：', err);
      }
    });
  }

  function pmRecordUnit() {
    const spec = PM_SPECS[$('pm-metric').value];
    const unit = spec ? spec.unit : 'ratio';
    $('pm-value-unit').textContent = unit === 'ratio' ? '%（如 91.5）' : PM_UNIT_TEXT[unit] || '';
    $('pm-value').max = unit === 'ratio' ? '100' : unit === 'seconds' ? '86400' : '720';
    return unit;
  }

  async function recordPlatformMetric(event) {
    event.preventDefault();
    const me = state.me;
    const status = $('pm-record-status');
    if (!isMerchant()) return;
    const metric = $('pm-metric').value;
    const day = $('pm-day').value;
    const raw = $('pm-value').value.trim();
    const sizeRaw = $('pm-size').value.trim();
    const unit = pmRecordUnit();
    let value = Number(raw);
    if (!metric || !day || !raw || !Number.isFinite(value) || value < 0) {
      setStatus(status, '请选指标、日期，并填写 0 以上的数值。', 'error');
      return;
    }
    if (unit === 'ratio') value = Math.round(value * 1000) / 100000; // 页面上填百分数，接口收 0–1
    const body = { metric, day, value };
    if (sizeRaw) {
      const size = Number(sizeRaw);
      if (!Number.isInteger(size) || size < 0) {
        setStatus(status, '样本量是 0 以上的整数（可以不填）。', 'error');
        return;
      }
      body.sample_size = size;
    }
    await busy(pmRecordCtx, [$('pm-record-submit')], async () => {
      try {
        const r = await api('POST', shopPath('platform-metrics'), body);
        if (state.me !== me) return;
        const what = `${label(Object.fromEntries(pmMetrics().map(([k, s]) => [k, s.label])), metric)} ${day}`;
        const message = !r.changed ? `${what}：和已登记的一样，没有改动` : r.replaced ? `已更新 ${what}` : `已登记 ${what}`;
        setStatus(status, message, 'ok');
        announce(message);
        $('pm-value').value = '';
        $('pm-size').value = '';
        loadPlatformMetrics();
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '登记没有完成：', err);
      }
    });
  }

  /** 在报表页加上「平台客服指标」一节（启动时建一次；内容随登录加载、退出时清空）。 */
  function initPlatformMetrics() {
    const panel = $('panel-reports');
    if (!panel) return;
    const windowSelect = h('select', { id: 'pm-window' });
    for (const days of PM_WINDOWS) {
      const option = h('option', { value: String(days) }, `各 ${days} 天`);
      if (days === PM_DEFAULT_WINDOW) option.defaultSelected = true;
      windowSelect.append(option);
    }
    const goLive = h('input', { id: 'pm-go-live', type: 'date', autocomplete: 'off' });
    goLive.addEventListener('input', () => delete goLive.dataset.auto);
    const form = h('form', { id: 'pm-form', className: 'inline-form pm-form', novalidate: true }, [
      field('pm-go-live', '接入日期', goLive),
      field('pm-window', '对比天数', windowSelect),
      h('button', { type: 'submit', id: 'pm-compare', className: 'secondary' }, '按接入日期对比'),
    ]);
    form.addEventListener('submit', (event) => {
      event.preventDefault();
      if (state.me) loadPlatformMetrics();
    });
    form.addEventListener('reset', () => delete goLive.dataset.auto);
    const sync = h('button', { type: 'button', id: 'pm-sync' }, '同步平台数据');
    sync.addEventListener('click', () => {
      if (state.me) syncPlatformMetrics();
    });

    const metric = h('select', { id: 'pm-metric' });
    for (const [key, spec] of pmMetrics()) metric.append(h('option', { value: key }, spec.label));
    metric.addEventListener('change', pmRecordUnit);
    const value = h('input', { id: 'pm-value', type: 'number', inputmode: 'decimal', min: '0', step: 'any' });
    const record = h('form', { id: 'pm-record-form', novalidate: true }, [
      h('div', { className: 'field-row' }, [
        field('pm-metric', '指标', metric),
        field('pm-day', '日期', h('input', { id: 'pm-day', type: 'date', autocomplete: 'off' })),
      ]),
      h('div', { className: 'field-row' }, [
        h('div', { className: 'field' }, [
          h('label', { for: 'pm-value' }, ['数值 ', h('span', { id: 'pm-value-unit', className: 'muted' })]),
          value,
        ]),
        field(
          'pm-size',
          '样本量（可不填）',
          h('input', { id: 'pm-size', type: 'number', inputmode: 'numeric', min: '0', step: '1' }),
        ),
      ]),
      h('button', { type: 'submit', id: 'pm-record-submit' }, '登记'),
      h('p', { id: 'pm-record-status', className: 'status', role: 'status' }),
    ]);
    record.addEventListener('submit', recordPlatformMetric);

    panel.append(
      h('section', { className: 'pm-section', 'aria-labelledby': 'pm-title' }, [
        h('h3', { id: 'pm-title' }, '平台客服指标 · 接入前后对比'),
        h(
          'p',
          { className: 'hint' },
          '平台考核商家客服的指标，接入智能辅助前后各看一段（§12.4）。同一天优先用平台数据，其次商家登记；' +
            '数据来自平台云内的接口，也可以从平台后台抄录登记。',
        ),
        form,
        h('div', { className: 'actions tight' }, [sync]),
        h('p', { id: 'pm-status', className: 'status', role: 'status' }),
        h('div', { id: 'platform-metrics', className: 'pm-report', 'data-dyn': true }),
        h('details', { id: 'pm-record', className: 'subpanel', 'data-write': true, hidden: true }, [
          h('summary', null, '登记平台后台的数据'),
          h('p', { className: 'hint' }, '从平台后台抄录某一天的数；同一指标同一天再登记会覆盖之前登记的（平台同步的数另存）。'),
          record,
        ]),
      ]),
    );
    pmRecordUnit();
  }

  // ================================================================ 启动

  const LOADERS = {
    knowledge: loadKnowledge,
    policies: loadPolicies,
    items: loadItemsTab,
    config: loadConfig,
    rollout: loadRollout,
    review: loadReviewTab,
    reports: loadReportsTab,
  };

  /** 正显示的页签原地刷新（stale）：不关掉正在编辑的表单。没列出的页签直接重新加载。 */
  const REFRESHERS = {
    // 列表里有打开的修改、确认面板（或焦点在列表里）时只刷新「要补的知识」，面板里写了一半的内容不丢
    knowledge: () => (busyEditing('#candidates, #faqs') ? loadGaps() : loadKnowledge()),
    config: refreshConfig,
    rollout: () => loadRollout(busyEditing('#stage-controls')),
  };

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
    fillOptions($('faq-status'), L.knowledgeStatus);
    const approved = $('faq-status').querySelector('option[value="approved"]');
    if (approved) approved.defaultSelected = true; // FAQ 列表默认只看已上线的（候选知识在上面单独列出）
    fillOptions($('faq-intent'), L.intents);
    fillOptions($('review-filter'), L.reviewStatus);
    if (LIMITS.question) $('faq-question').maxLength = LIMITS.question;
    if (LIMITS.answer) $('faq-answer').maxLength = LIMITS.answer;
    if (LIMITS.sample_size_max) $('review-size').max = String(LIMITS.sample_size_max);
    $('review-size').defaultValue = String(LIMITS.sample_size || 50); // 表单 reset 后回到默认条数
    initPlatformMetrics(); // 报表页的「平台客服指标」一节（§12.4）

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
