/*
 * Rodit 售后审核工作台（方案 §7.3.2 商家侧审核助手、§15 Phase 2、§2.1 售后平均审核时长）。
 *
 * 约定（和话术助手、商家控制台两个页面的脚本一样）：
 * - 服务端和用户给的文字一律用 textContent / 文本节点写进页面，从不按 HTML 解析（防 XSS）；
 * - 令牌只放 sessionStorage，只放进 Authorization 请求头，不进网址、不打日志；
 *   和另外两个页面用同一个键，同一标签页里互相跳转不用重新登录；
 * - 接口地址用相对路径；店铺接口一律写成 shopPath('字面路径', 查询参数)，测试据此核对接口清单；
 * - 每次登录换一个 state.me 对象：响应回来时已经退出或换人登录就丢掉，不写进下一位的界面；
 *   401 只在发请求的令牌还是当前令牌时才登出；
 * - 退出、令牌无效时只在 sessionStorage 里存着的还是本页的令牌时才清（forgetToken）；从往返缓存
 *   恢复时存着的令牌变了，先丢掉本页内存里的身份，有令牌再用它登录；
 * - 同一块区域的加载按请求编号只用最近一次的结果（nextRun / latest）：换筛选、换售后单后，
 *   晚回来的旧响应不覆盖新的；防重复提交的 busy 记下发起的身份，退出后不影响下一位的按钮；
 * - 写操作（同意 / 拒绝）只在客服在确认对话框里点「确认执行」后才发（§4.1 原则 3），带上详情里
 *   平台当前的状态版本（expected_status_version）；状态已变化时提示刷新并自动刷新；
 * - 按钮能不能点只是界面提示（详情接口给出原因），真正的检查在接口和审核助手里（§9.2、§7.3.2）。
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
    states: CONFIG.review_states || {},
    advice: CONFIG.advice || {},
    types: CONFIG.types || {},
    reasons: CONFIG.reasons || {},
    statuses: CONFIG.statuses || {},
    actions: CONFIG.actions || {},
    nextSteps: CONFIG.next_steps || {},
  };
  const RULES = CONFIG.rules || {};
  const GROUPS = CONFIG.role_groups || {};
  const FLOW = Array.isArray(CONFIG.state_flow) ? CONFIG.state_flow : [];
  const LIMITS = CONFIG.limits || {};
  const UNREVIEWED = CONFIG.unreviewed || 'unreviewed';
  const DECIDED = ['agreed', 'refused', 'wait_return']; // 客服确认执行后进入的状态

  const STORAGE_TOKEN = 'rodit.assist.token'; // 三个页面同一个键（同一标签页里切换页面不用重新登录）
  const STORAGE_SESSION = 'rodit.assist.session'; // 话术助手页的会话编号：退出时一并清掉
  const STORAGE_TAB = 'rodit.console.tab'; // 控制台的页签偏好：退出时一并清掉

  const state = {
    token: null,
    stored: null, // 本页写进 sessionStorage 的令牌（退出时只清自己的；存不了时为 null）
    me: null,
    clockSkew: 0, // 服务端时钟减本机时钟（毫秒），登录时按 /v1/me 的 server_time 算
    signingIn: false,
    selected: null, // 正在看的售后单 ID
    detail: null, // 最近一次的详情
    pending: null, // 确认对话框里待执行的操作 {id, action, version, needsCode, opener}
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
        else if (key === 'disabled') node.disabled = true;
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

  /** 把登录用的令牌存进 sessionStorage，记下是本页存的。 */
  function keepToken(token) {
    storage.set(STORAGE_TOKEN, token);
    state.stored = storage.get(STORAGE_TOKEN) === token ? token : null;
  }

  /** 清掉 sessionStorage 里的令牌和另外两个页面的状态——只在存着的还是这个令牌时才清。 */
  function forgetToken(token) {
    if (!token || storage.get(STORAGE_TOKEN) !== token) return;
    storage.remove(STORAGE_TOKEN);
    storage.remove(STORAGE_SESSION);
    storage.remove(STORAGE_TAB);
  }

  const enc = encodeURIComponent;
  const has = (map, key) => Object.prototype.hasOwnProperty.call(map, key);
  const inGroup = (group, role) => Array.isArray(GROUPS[group]) && GROUPS[group].includes(role);
  const hasText = (value) => typeof value === 'string' && value.trim() !== '';

  let seq = 0;
  const uid = (prefix) => `${prefix}-${(seq += 1)}`;

  function label(map, key) {
    return has(map, key) ? map[key] : String(key);
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

  /** 时间 → 东八区的 "2026-09-28 10:35"。 */
  function fmtTime(value) {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    if (!TIME_FORMAT) return date.toISOString().replace('T', ' ').slice(0, 16);
    const p = {};
    for (const part of TIME_FORMAT.formatToParts(date)) p[part.type] = part.value;
    return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}`;
  }

  /** 秒 → "3 小时 20 分""2 天 4 小时""15 分钟"。 */
  function fmtDuration(seconds) {
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '—';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 1) return '不到 1 分钟';
    if (minutes < 60) return `${minutes} 分钟`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return minutes % 60 ? `${hours} 小时 ${minutes % 60} 分` : `${hours} 小时`;
    const days = Math.floor(hours / 24);
    return hours % 24 ? `${days} 天 ${hours % 24} 小时` : `${days} 天`;
  }

  function money(amount) {
    if (amount === null || amount === undefined || amount === '') return '—';
    const n = Number(amount);
    return Number.isFinite(n) ? `¥${n.toFixed(2)}` : `¥${amount}`;
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

  function reportFailure(statusNode, prefix, err) {
    const message = `${prefix}${err.message}`;
    setStatus(statusNode, message, 'error');
    showError(message);
  }

  /**
   * 防重复提交：同一个 ctx 上一次还没完成时直接返回；执行期间禁用 controls，结束后恢复原来的状态
   * （本来就不能点的按钮——比如不满足条件的操作——结束后还是不能点）。
   * ctx 记下发起的身份和第几次：退出或换人登录后，上一位没完成的那次不挡下一位。
   */
  async function busy(ctx, controls, task) {
    if (ctx.busy && ctx.owner === state.me) return undefined;
    const run = (ctx.run = (ctx.run || 0) + 1);
    ctx.busy = true;
    ctx.owner = state.me;
    const list = controls.filter(Boolean).map((control) => [control, control.disabled]);
    for (const [control] of list) control.disabled = true;
    try {
      return await task();
    } finally {
      if (ctx.run === run) {
        ctx.busy = false;
        ctx.owner = null;
        for (const [control, was] of list) if (control.isConnected) control.disabled = was;
      }
    }
  }

  const sharedBusy = [];

  function sharedCtx(...controlIds) {
    const ctx = { busy: false, controlIds };
    sharedBusy.push(ctx);
    return ctx;
  }

  function resetBusy() {
    for (const ctx of sharedBusy) {
      ctx.run = (ctx.run || 0) + 1;
      ctx.busy = false;
      ctx.owner = null;
      for (const id of ctx.controlIds) $(id).disabled = false;
    }
  }

  const runs = {};

  function nextRun(key) {
    runs[key] = (runs[key] || 0) + 1;
    return runs[key];
  }

  const latest = (key, run) => runs[key] === run;

  function dropRuns() {
    for (const key of Object.keys(runs)) nextRun(key);
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

  function fillOptions(select, entries) {
    for (const [value, text] of entries) select.append(h('option', { value }, text));
  }

  // ================================================================ 接口

  class ApiFailure extends Error {
    constructor(status, code, message, data) {
      super(message);
      this.status = status;
      this.code = code;
      this.data = data || null;
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

  /** 售后单状态已变化：确认时平台重读到的状态版本和客服看到的不一样（§7.3.2 防止误操作）。 */
  function isStale(err) {
    if (!err) return false;
    if (err.code === 'conflict') return true;
    return err.code === 'action_not_allowed' && /状态已变化/.test(err.message || '');
  }

  // ================================================================ 登录

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
      if (run !== signIns) return false;
      state.token = null;
      if (err.status === 401 || err.status === 403) forgetToken(token);
      if (err.code === 'forbidden_role') {
        showError('这个令牌是平台对接用的，不能登录网页工具；请用客服、店长或商家的令牌。');
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
    if (!inGroup('desk', me.role)) {
      // 复核人员：售后审核不归复核人员做。令牌留着，同一标签页打开控制台时直接登录
      state.token = null;
      keepToken(token);
      $('token').value = '';
      showIdentity(me);
      showError(`${label(L.roles, me.role)}请使用商家控制台，售后审核只给客服、店长和商家用。`, {
        href: 'console',
        text: '打开商家控制台',
      });
      return false;
    }
    state.me = me;
    state.clockSkew = clockSkewOf(me);
    keepToken(token);
    enterApp();
    return true;
  }

  function clockSkewOf(me) {
    const t = Date.parse(me && me.server_time);
    return Number.isFinite(t) ? t - Date.now() : 0;
  }

  function showIdentity(me) {
    $('whoami').hidden = false;
    $('identity').textContent = `${label(L.roles, me.role)} ${me.user_id} · 店铺 ${me.shop_id}`;
  }

  function enterApp() {
    $('token').value = '';
    $('login').hidden = true;
    $('app').hidden = false;
    showIdentity(state.me);
    $('detail-empty').hidden = false;
    loadStats();
    loadQueue();
    $('run-rules').focus();
    announce(`已登录：${$('identity').textContent}`);
  }

  /**
   * 退出：清掉本页的身份、数据和表单。sessionStorage 只在存着的还是本页的令牌时才清（forgetToken）；
   * opts.keepStorage：只清本页（后退恢复时发现另一个页面已经退出或换人登录）。
   */
  function signOut(opts = {}) {
    if (!opts.keepStorage) forgetToken(state.stored);
    signIns += 1;
    state.signingIn = false;
    $('login-button').disabled = false;
    state.token = null;
    state.stored = null;
    state.me = null;
    state.selected = null;
    state.detail = null;
    closeConfirm();
    dropRuns();
    resetBusy();
    for (const node of document.querySelectorAll('[data-dyn]')) put(node);
    for (const node of document.querySelectorAll('#app .status')) setStatus(node, '', '');
    $('detail-empty').hidden = false;
    for (const form of $('app').querySelectorAll('form')) form.reset();
    $('app').hidden = true;
    $('whoami').hidden = true;
    $('identity').textContent = '';
    $('login').hidden = false;
    $('token').focus();
  }

  /** 从往返缓存恢复：离开期间另一个页面退出或换人登录了，先丢掉本页的身份，有令牌再用它登录。 */
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

  // ================================================================ 概况（§2.1 售后平均审核时长）

  function sinceParam() {
    const days = Number($('stats-range').value);
    if (!days) return null;
    const now = Date.now() + (state.clockSkew || 0);
    return new Date(now - days * 86400000).toISOString();
  }

  async function loadStats() {
    const me = state.me;
    const run = nextRun('stats');
    loading($('stats'));
    try {
      const stats = await api('GET', shopPath('aftersales/stats', { since: sinceParam() }));
      if (state.me !== me || !latest('stats', run)) return;
      renderStats(stats);
    } catch (err) {
      if (state.me !== me || !latest('stats', run)) return;
      loadFailed($('stats'), '概况', err);
    }
  }

  function tile(name, value, note, hot) {
    return h('li', { className: hot ? 'tile hot' : 'tile' }, [
      h('span', { className: 'tile-label' }, name),
      h('span', { className: 'tile-value' }, value),
      note ? h('span', { className: 'tile-note' }, note) : null,
    ]);
  }

  function renderStats(s) {
    const by = s.by_state || {};
    const n = (key) => (typeof by[key] === 'number' ? by[key] : 0);
    const d = s.review_duration || {};
    const w = s.waiting || {};
    const range = $('stats-range').selectedOptions[0];
    const rangeText = range ? range.textContent : '';
    const durationNote = d.count
      ? `${rangeText} ${d.count} 单 · 中位 ${fmtDuration(d.median_s)} · P90 ${fmtDuration(d.p90_s)}`
      : `${rangeText}还没有在工作台确认执行的售后单`;
    const closed = n('agreed') + n('refused') + n('platform');
    const tiles = h('ul', { className: 'tiles', 'aria-label': '售后审核概况' }, [
      tile('还没结束', String(s.open_total || 0), `其中 ${s.needs_action || 0} 张要处理`, (s.needs_action || 0) > 0),
      tile(label(L.nextSteps, 'run_rules'), String(n(UNREVIEWED))),
      tile(label(L.states, 'confirm'), String(n('confirm'))),
      tile(label(L.states, 'verify'), String(n('verify'))),
      tile(label(L.states, 'evidence'), String(n('evidence'))),
      tile(label(L.states, 'negotiate'), String(n('negotiate'))),
      tile(label(L.states, 'wait_return'), String(n('wait_return'))),
      tile('平均审核时长', d.count ? fmtDuration(d.average_s) : '—', durationNote),
      tile('最久已等待', w.count ? fmtDuration(w.max_s) : '—', w.count ? `还没结束的 ${w.count} 张` : null),
      tile(
        `已结束（${rangeText}）`,
        String(closed),
        `${label(L.states, 'agreed')} ${n('agreed')} · ${label(L.states, 'refused')} ${n('refused')} · ${label(L.states, 'platform')} ${n('platform')}`,
      ),
    ]);
    const rules = Object.entries(s.pending_by_rule || {}).filter(([, count]) => count > 0);
    const ruleLine = rules.length
      ? h('p', { className: 'rule-counts hint' }, [
          '还没结束的售后单命中的规则：',
          rules.map(([rule, count], i) => [
            i ? ' · ' : '',
            h('span', { className: 'rule-chip', title: ruleText(rule) }, rule),
            ` ${count} 张`,
          ]),
        ])
      : null;
    const notes = [];
    if (s.duration_unknown) notes.push(h('p', { className: 'hint' }, `${s.duration_unknown} 单读不到平台申请时间，没算进审核时长。`));
    if (s.platform_unavailable) {
      notes.push(h('p', { className: 'status warn' }, '平台待审核列表暂时读不到：未跑规则的单数不准（§12.2）。'));
    }
    put(
      $('stats'),
      tiles,
      ruleLine,
      h('p', { className: 'hint' }, '平均审核时长：售后申请 → 在工作台第一次确认执行（抖店「售后平均审核时长」的近似，§2.1）；商家直接在平台后台处理的不计入。'),
      notes,
    );
  }

  function ruleText(rule) {
    const info = RULES[rule];
    return info ? `${rule}：${info.condition} → ${info.advice}` : rule;
  }

  // ================================================================ 待审核列表

  async function loadQueue() {
    const me = state.me;
    const run = nextRun('queue');
    const list = $('queue');
    loading(list);
    try {
      const queue = await api(
        'GET',
        shopPath('aftersales', {
          state: $('filter-state').value,
          status: $('filter-status').value,
          rule: $('filter-rule').value,
        }),
      );
      if (state.me !== me || !latest('queue', run)) return;
      renderQueue(queue);
    } catch (err) {
      if (state.me !== me || !latest('queue', run)) return;
      put($('queue-summary'));
      loadFailed(list, '待审核列表', err);
    }
  }

  function renderQueue(q) {
    const rows = Array.isArray(q.rows) ? q.rows : [];
    const summary = [`共 ${q.total || 0} 张，要处理 ${q.needs_action || 0} 张，未跑规则 ${q.unreviewed || 0} 张`];
    if (rows.length < (q.total || 0)) summary.push(`（只显示前 ${rows.length} 张）`);
    put(
      $('queue-summary'),
      summary.join(''),
      q.platform_unavailable ? h('span', { className: 'warn-text' }, '。平台待审核列表暂时读不到，只列出了已跑过规则的售后单（§12.2）') : null,
    );
    if (!rows.length) {
      put($('queue'), h('li', { className: 'muted' }, '没有符合条件的售后单。'));
      return;
    }
    put($('queue'), rows.map(queueItem));
  }

  function queueItem(row) {
    const id = row.aftersale_id;
    const step = row.next_step || 'done';
    const metaParts = [];
    if (row.type) metaParts.push(label(L.types, row.type));
    if (row.reason) metaParts.push(label(L.reasons, row.reason));
    if (row.amount !== null && row.amount !== undefined) metaParts.push(money(row.amount));
    if (row.platform_status) metaParts.push(`平台：${label(L.statuses, row.platform_status)}`);
    const time = [];
    if (row.created_at) time.push(`申请于 ${fmtTime(row.created_at)}`);
    if (typeof row.waiting_s === 'number') time.push(`已等待 ${fmtDuration(row.waiting_s)}`);
    const classes = ['queue-item'];
    if (row.needs_action) classes.push('act');
    if (step === 'done') classes.push('done');
    const button = h(
      'button',
      {
        type: 'button',
        className: classes.join(' '),
        'data-id': id,
        'aria-current': state.selected === id ? 'true' : 'false',
        onclick: () => selectRow(id, true),
      },
      [
        h('span', { className: 'qi-line' }, [
          h('span', { className: 'qi-id' }, id),
          stateBadges(row.reviewed ? row.state : null, step),
          row.requires_manager ? h('span', { className: 'badge manager' }, '须店长确认') : null,
          Array.isArray(row.risk_flags) && row.risk_flags.length ? h('span', { className: 'badge risk' }, '风险') : null,
        ]),
        metaParts.length ? h('span', { className: 'qi-line qi-meta' }, metaParts.join(' · ')) : null,
        time.length ? h('span', { className: 'qi-line qi-meta' }, time.join(' · ')) : null,
        row.suggestion || (row.matched_rules || []).length
          ? h('span', { className: 'qi-line qi-meta' }, [
              row.suggestion ? label(L.advice, row.suggestion) : '',
              (row.matched_rules || []).length ? `（${row.matched_rules.join('、')}）` : '',
            ])
          : null,
        row.platform_changed ? h('span', { className: 'qi-line warn-text' }, '平台上的售后单变了，刷新后再处理') : null,
        row.platform_unavailable ? h('span', { className: 'qi-line warn-text' }, '平台数据暂时读不到') : null,
      ],
    );
    return h('li', null, button);
  }

  /**
   * 审核状态的标记（没跑规则时是「未跑规则」），鼠标悬停看下一步；要刷新时另加「需要刷新」。
   * 颜色按下一步：要处理的（待客服确认、未跑规则、需要刷新）醒目，在等别人的、已结束的平淡。
   */
  function stateBadges(reviewState, step) {
    const text = reviewState ? label(L.states, reviewState) : label(L.nextSteps, 'run_rules');
    const cls = ['badge', `step-${step}`];
    if (reviewState) cls.push(`st-${reviewState}`);
    return [
      h('span', { className: cls.join(' '), title: label(L.nextSteps, step) }, text),
      step === 'refresh' ? h('span', { className: 'badge step-refresh' }, label(L.nextSteps, step)) : null,
    ];
  }

  function markSelected() {
    for (const node of document.querySelectorAll('#queue .queue-item')) {
      node.setAttribute('aria-current', node.dataset.id === state.selected ? 'true' : 'false');
    }
  }

  // ================================================================ 详情

  function selectRow(id, focus) {
    if (!state.me) return;
    state.selected = id;
    markSelected();
    setStatus($('detail-status'), '', '');
    loadDetail(id, { focus });
  }

  async function loadDetail(id, opts = {}) {
    const me = state.me;
    const run = nextRun('detail');
    $('detail-empty').hidden = true;
    loading($('detail'));
    try {
      const detail = await api('GET', shopPath(`aftersales/${enc(id)}`));
      if (state.me !== me || !latest('detail', run) || state.selected !== id) return null;
      state.detail = detail;
      renderDetail(detail);
      if (opts.focus) $('detail-panel').focus();
      return detail;
    } catch (err) {
      if (state.me !== me || !latest('detail', run)) return null;
      state.detail = null;
      loadFailed($('detail'), `售后单 ${id} 的详情`, err);
      return null;
    }
  }

  function visitedStates(review) {
    const seen = new Set();
    for (const step of (review && review.history) || []) {
      seen.add(step.from_state);
      seen.add(step.to_state);
    }
    return seen;
  }

  function stateStrip(review) {
    const current = review ? review.state : 'new';
    const seen = visitedStates(review);
    return h(
      'ol',
      { className: 'state-strip', 'aria-label': '审核进度（§7.3.2 状态机）' },
      FLOW.map((s) => {
        const isCurrent = s === current;
        const cls = isCurrent ? 'current' : seen.has(s) ? 'visited' : null;
        return h(
          'li',
          { className: cls, 'aria-current': isCurrent ? 'step' : null, 'data-state': s },
          [label(L.states, s), isCurrent ? h('span', { className: 'visually-hidden' }, '（当前）') : seen.has(s) ? h('span', { className: 'visually-hidden' }, '（已经过）') : null],
        );
      }),
    );
  }

  function factsList(d) {
    const p = d.platform;
    const t = d.ticket;
    const rows = [];
    const add = (name, value, className) => rows.push(h('dt', null, name), h('dd', { className }, value));
    if (t) add('订单', t.order_id);
    if (p) {
      add('类型', label(L.types, p.type));
      add('原因', label(L.reasons, p.reason));
      add('平台状态', `${label(L.statuses, p.status)} · 状态版本 ${p.status_version}`);
      add('金额', money(p.amount));
      add('凭证', `${p.evidence_count || 0} 张`);
    } else if (d.review) {
      add('平台状态', `${label(L.statuses, d.review.platform_status)}（审核记录上次同步到的）`);
    }
    if (t) {
      add('申请时间', fmtTime(t.created_at));
      if (typeof t.waiting_s === 'number') add('已等待', fmtDuration(t.waiting_s));
      if (hasText(t.note)) add('买家说明', t.note, 'buyer-note');
    }
    return h('dl', { className: 'facts' }, rows);
  }

  function renderDetail(d) {
    const review = d.review;
    const id = d.aftersale_id;
    const parts = [];
    parts.push(
      h('div', { className: 'detail-head' }, [
        h('h3', null, `售后单 ${id}`),
        stateBadges(review ? review.state : null, d.next_step),
      ]),
      h('p', { className: 'hint next-step' }, `下一步：${label(L.nextSteps, d.next_step)}`),
    );
    if (d.platform_unavailable) {
      const asOf = d.ticket && d.ticket.fetched_at ? `（截至 ${fmtTime(d.ticket.fetched_at)}）` : '';
      parts.push(h('p', { className: 'stale-note' }, `平台接口暂时不可用：显示的是缓存数据${asOf}，现在不能确认，稍后刷新。`));
    }
    parts.push(factsList(d));
    parts.push(h('h4', null, '审核进度'), stateStrip(review));

    if (review) {
      parts.push(
        h('div', { className: 'suggestion' }, [
          h('p', { className: 'suggestion-advice' }, `规则建议：${review.suggestion ? label(L.advice, review.suggestion) : '—'}`),
          hasText(review.reason) ? h('p', null, review.reason) : null,
          review.verify_started_at ? h('p', { className: 'hint' }, `签收核查开始于 ${fmtTime(review.verify_started_at)}`) : null,
        ]),
      );
      const rules = review.matched_rules || [];
      if (rules.length) {
        parts.push(
          h('h4', null, '命中的规则'),
          h(
            'dl',
            { className: 'rules' },
            rules.map((rule) => {
              const info = RULES[rule];
              return [
                h('dt', null, h('span', { className: 'rule-chip' }, rule)),
                h('dd', null, info ? `${info.condition} → ${info.advice}` : rule),
              ];
            }),
          ),
        );
      }
      if ((review.risk_flags || []).length) {
        parts.push(h('ul', { className: 'risks', 'aria-label': '风险标记' }, review.risk_flags.map((f) => h('li', null, f))));
      }
      if (review.needs_refresh) {
        parts.push(h('p', { className: 'stale-note' }, '平台上的售后单在给出建议之后变了，建议可能过期：请先刷新再确认。'));
      }
    } else {
      parts.push(h('p', { className: 'muted' }, '还没跑规则：点「跑规则」拉取售后单并给出建议（只读，不执行操作）。'));
    }

    if (d.manager_required) {
      parts.push(
        h(
          'p',
          { className: 'manager-note', id: 'manager-note' },
          d.caller_is_manager
            ? '金额超过店铺上限（R6）：必须店长确认，你是店长，可以确认。'
            : '金额超过店铺上限，必须店长确认（R6）。当前账号不能确认，请交给店长处理。',
        ),
      );
    }

    parts.push(h('h4', null, '操作'));
    if (!d.write_enabled) {
      parts.push(h('p', { className: 'write-note' }, '本单元没有接售后写接口（MVP 只给建议，§8.4）：请按建议在平台后台操作。'));
    } else if (!d.write_scope) {
      parts.push(h('p', { className: 'write-note' }, '店铺授权没有写售后权限（refund.write，二期才申请）：请按建议在平台后台操作。'));
    }
    parts.push(actionList(d));
    if (hasText(d.refuse_hint)) parts.push(h('p', { className: 'hint' }, d.refuse_hint));

    if (review && (review.history || []).length) {
      parts.push(
        h('h4', null, '处理记录'),
        h(
          'ol',
          { className: 'history' },
          review.history.map((step) =>
            h('li', null, [
              `${fmtTime(step.at)} ${label(L.states, step.from_state)} → ${label(L.states, step.to_state)}`,
              step.by ? ` · ${step.by}` : '',
              hasText(step.note) ? ` · ${step.note}` : '',
            ]),
          ),
        ),
      );
    }
    if (review && review.last_result) {
      const r = review.last_result;
      const name = label(L.actions, r.action);
      parts.push(
        h(
          'p',
          { className: `result-line ${r.ok ? 'ok' : 'bad'}` },
          r.ok
            ? `最近一次操作：${name}成功（平台返回 ${r.platform_code || '—'}）${r.duplicate ? '，重复点击没有重复执行' : ''}`
            : `最近一次操作：${name}没有成功：${r.message || r.platform_code || '平台没有说明原因'}`,
        ),
      );
    }
    put($('detail'), parts);
  }

  // 详情里的按钮共用一个防重复提交标记（列表刷新、换售后单时按钮会重建）
  const detailCtx = { busy: false };

  function detailButtons() {
    return Array.from(document.querySelectorAll('#detail button'));
  }

  function actionList(d) {
    const items = [];
    for (const option of d.actions || []) {
      const whyId = uid('why');
      const why = option.available
        ? option.needs_reason_code
          ? '需要填写拒绝原因码；确认后执行，执行前会重读平台状态'
          : '确认后执行，执行前会重读平台状态'
        : option.blocked_reason || '现在不能执行';
      const button = h(
        'button',
        {
          type: 'button',
          className: option.recommended ? 'recommended' : null,
          'data-action': option.action,
          'aria-describedby': whyId,
          disabled: !option.available,
          onclick: (event) => openConfirm(d, option, event.currentTarget),
        },
        `${option.label}${option.recommended ? '（建议）' : ''}`,
      );
      items.push(h('li', { className: 'action-row' }, [button, h('p', { className: 'why', id: whyId }, why)]));
    }
    const id = d.aftersale_id;
    if (d.can_negotiate) {
      items.push(entryRow('协商', '协商备注（选填）', '把规则建议之外的处理方式先和买家谈（§7.3.2；淘宝拒绝前要先协商）', (text) => negotiate(id, text)));
    }
    if (d.can_finish_negotiation) {
      items.push(entryRow('协商有结果', '协商结果（选填）', '回到待客服确认；平台上已经处理的会同步状态', (text) => negotiationDone(id, text)));
    }
    if (d.can_run_rules) {
      const whyId = uid('why');
      items.push(
        h('li', { className: 'action-row' }, [
          h('button', { type: 'button', 'aria-describedby': whyId, onclick: () => reviewOne(id, '已跑规则') }, '跑规则'),
          h('p', { className: 'why', id: whyId }, '拉取售后单、订单、物流，跑 R1–R8 给出建议（只读）'),
        ]),
      );
    }
    if (d.can_refresh) {
      const whyId = uid('why');
      items.push(
        h('li', { className: 'action-row' }, [
          h('button', { type: 'button', className: 'secondary', 'aria-describedby': whyId, onclick: () => reviewOne(id, '已刷新') }, '刷新'),
          h('p', { className: 'why', id: whyId }, '重读平台状态、推进状态机（签收核查满时、凭证补充、买家寄回、平台已处理）'),
        ]),
      );
    }
    return h('ul', { className: 'action-list', 'aria-label': '可以做的操作' }, items);
  }

  /** 一个带备注框的操作（协商、协商有结果）：点按钮展开备注框，再点「提交」。 */
  function entryRow(name, fieldLabel, hint, submit) {
    const whyId = uid('why');
    const fieldId = uid('entry');
    const panel = h('div', { className: 'entry-form', hidden: true });
    const textarea = h('textarea', { id: fieldId, rows: '2', maxlength: String(LIMITS.note || 500) });
    const send = h('button', { type: 'button' }, `提交${name}`);
    const cancel = h('button', { type: 'button', className: 'secondary' }, '取消');
    const toggle = h('button', { type: 'button', className: 'secondary', 'aria-describedby': whyId, 'aria-expanded': 'false' }, name);
    const close = () => {
      panel.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
      toggle.focus();
    };
    toggle.addEventListener('click', () => {
      panel.hidden = !panel.hidden;
      toggle.setAttribute('aria-expanded', panel.hidden ? 'false' : 'true');
      if (!panel.hidden) textarea.focus();
    });
    cancel.addEventListener('click', close);
    panel.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') close();
    });
    send.addEventListener('click', () => submit(textarea.value.trim()));
    panel.append(
      h('div', { className: 'field' }, [h('label', { for: fieldId }, fieldLabel), textarea, h('p', { className: 'hint' }, '会打码后留痕；不要写买家手机号、地址。')]),
      h('div', { className: 'actions tight' }, [send, cancel]),
    );
    return h('li', null, [h('div', { className: 'action-row' }, [toggle, h('p', { className: 'why', id: whyId }, hint)]), panel]);
  }

  /** 详情里的操作结束后：重新加载详情、列表和概况（详情接口给出新的状态和能做的操作）。 */
  async function afterChange(id) {
    loadQueue();
    loadStats();
    if (state.selected === id) await loadDetail(id);
  }

  async function reviewOne(id, done) {
    const me = state.me;
    const status = $('detail-status');
    await busy(detailCtx, detailButtons(), async () => {
      setStatus(status, '正在读取平台售后单…', 'muted');
      try {
        const review = await api('POST', shopPath(`aftersales/${enc(id)}/review`), {});
        if (state.me !== me) return;
        setStatus(status, `${done}：${label(L.states, review.state)}${review.suggestion ? `，${label(L.advice, review.suggestion)}` : ''}`, 'ok');
        announce(status.textContent);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有完成：', err);
      }
    });
    if (state.me === me) await afterChange(id);
  }

  async function negotiate(id, note) {
    const me = state.me;
    const status = $('detail-status');
    await busy(detailCtx, detailButtons(), async () => {
      try {
        await api('POST', shopPath(`aftersales/${enc(id)}/negotiate`), note ? { note } : {});
        if (state.me !== me) return;
        setStatus(status, '已进入协商：和买家谈出结果后，回来点「协商有结果」。', 'ok');
        announce(status.textContent);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有进入协商：', err);
      }
    });
    if (state.me === me) await afterChange(id);
  }

  async function negotiationDone(id, outcome) {
    const me = state.me;
    const status = $('detail-status');
    await busy(detailCtx, detailButtons(), async () => {
      try {
        const review = await api('POST', shopPath(`aftersales/${enc(id)}/negotiation-done`), outcome ? { outcome } : {});
        if (state.me !== me) return;
        setStatus(status, `协商已记录：${label(L.states, review.state)}。`, 'ok');
        announce(status.textContent);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有记录协商结果：', err);
      }
    });
    if (state.me === me) await afterChange(id);
  }

  // ================================================================ 确认执行（§4.1 原则 3）

  function openConfirm(d, option, opener) {
    if (!state.me || !option.available || !d.platform) return;
    const review = d.review;
    state.pending = {
      id: d.aftersale_id,
      action: option.action,
      version: d.platform.status_version,
      needsCode: Boolean(option.needs_reason_code),
      opener,
    };
    const facts = [];
    const add = (name, value) => facts.push(h('dt', null, name), h('dd', null, value));
    add('操作', option.label);
    add('售后单', d.aftersale_id);
    add('金额', money(d.platform.amount));
    add('平台状态', `${label(L.statuses, d.platform.status)}（状态版本 ${d.platform.status_version}）`);
    if (review) add('规则建议', `${review.suggestion ? label(L.advice, review.suggestion) : '—'}（${(review.matched_rules || []).join('、') || '—'}）`);
    add('确认人', `${label(L.roles, state.me.role)} ${state.me.user_id}`);
    $('confirm-title').textContent = `确认${option.label}`;
    $('confirm-rule').textContent = CONFIG.confirm_rule || '售后写操作必须由人确认。';
    $('confirm-facts').replaceChildren(...facts);
    const refuse = option.action === 'refuse';
    $('refuse-fields').hidden = !refuse;
    $('reason-code-field').hidden = !state.pending.needsCode;
    $('reason-code').value = '';
    $('reason-text').value = '';
    setStatus($('confirm-status'), '', '');
    const dialog = $('confirm-dialog');
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
    (state.pending.needsCode ? $('reason-code') : $('confirm-cancel')).focus();
  }

  function closeConfirm() {
    const dialog = $('confirm-dialog');
    const opener = state.pending && state.pending.opener;
    state.pending = null;
    if (dialog.open) {
      if (typeof dialog.close === 'function') dialog.close();
      else dialog.removeAttribute('open');
    }
    if (opener && opener.isConnected) opener.focus();
  }

  const confirmCtx = sharedCtx('confirm-go', 'confirm-cancel');

  async function doConfirm() {
    const pending = state.pending;
    if (!pending || !state.me) return;
    const me = state.me;
    const status = $('confirm-status');
    const body = { action: pending.action, expected_status_version: pending.version };
    if (pending.action === 'refuse') {
      const code = $('reason-code').value.trim();
      const text = $('reason-text').value.trim();
      if (pending.needsCode && !code) {
        setStatus(status, '请填写拒绝原因码。', 'error');
        $('reason-code').focus();
        return;
      }
      if (code) body.reason_code = code;
      if (text) body.reason_text = text;
    }
    let outcome = null;
    await busy(confirmCtx, [$('confirm-go'), $('confirm-cancel')], async () => {
      setStatus(status, '正在重读平台状态并执行…', 'muted');
      try {
        const review = await api('POST', shopPath(`aftersales/${enc(pending.id)}/confirm`), body);
        if (state.me !== me) return;
        const result = review.last_result;
        if (result && !result.ok && result.platform_code === 'STATUS_CHANGED') outcome = { stale: true };
        else outcome = { review };
      } catch (err) {
        if (state.me !== me) return;
        if (isStale(err)) outcome = { stale: true };
        else reportFailure(status, '没有执行：', err);
      }
    });
    if (state.me !== me || !outcome || state.pending !== pending) return;
    closeConfirm();
    const detailStatus = $('detail-status');
    if (outcome.stale) {
      await staleRefresh(pending.id);
      return;
    }
    const review = outcome.review;
    const name = label(L.actions, pending.action);
    if (DECIDED.includes(review.state)) {
      setStatus(detailStatus, `已执行${name}：${label(L.states, review.state)}。`, 'ok');
    } else if (review.state === 'platform') {
      setStatus(detailStatus, `售后单在平台上已经处理，没有执行${name}（R1，只同步状态）。`, 'warn');
    } else {
      const r = review.last_result;
      const why = r && !r.ok ? `：${r.message || r.platform_code || '平台没有说明原因'}` : '';
      setStatus(detailStatus, `${name}没有成功${why}。`, 'error');
    }
    announce(detailStatus.textContent);
    await afterChange(pending.id);
  }

  /** 状态已变化（§7.3.2 调用前重新读取）：提示、自动刷新（重读平台、重新给建议），客服核对后重新确认。 */
  async function staleRefresh(id) {
    const me = state.me;
    const status = $('detail-status');
    const message = '售后单状态已变化，请刷新后重新确认。';
    setStatus(status, `${message}正在刷新…`, 'warn');
    showError(message);
    try {
      await api('POST', shopPath(`aftersales/${enc(id)}/review`), {});
    } catch (err) {
      if (state.me !== me) return;
      reportFailure(status, `${message}刷新没有完成：`, err);
      await afterChange(id);
      return;
    }
    if (state.me !== me) return;
    await afterChange(id);
    if (state.me !== me) return;
    setStatus(status, `${message}已刷新，请核对最新的建议和平台状态。`, 'warn');
    announce(status.textContent);
  }

  // ================================================================ 批量跑规则

  const runCtx = sharedCtx('run-rules');

  async function runRules() {
    if (!state.me) return;
    const me = state.me;
    const status = $('run-status');
    await busy(runCtx, [$('run-rules')], async () => {
      setStatus(status, '正在拉取售后单并跑规则…', 'muted');
      try {
        const result = await api('POST', shopPath('aftersales/review-pending'), { limit: LIMITS.batch || 20 });
        if (state.me !== me) return;
        if (!result.attempted) {
          setStatus(status, '没有需要跑规则的售后单。', 'ok');
        } else {
          const failures = (result.rows || []).filter((r) => !r.ok).map((r) => `${r.aftersale_id}（${r.message || r.error}）`);
          const parts = [`跑了 ${result.attempted} 张：成功 ${result.succeeded} 张`];
          if (result.failed) parts.push(`，失败 ${result.failed} 张：${failures.join('；')}`);
          if (result.remaining) parts.push(`；还有 ${result.remaining} 张没轮到，再点一次继续`);
          if (result.stopped) parts.push(`；中途停下：${result.stopped}`);
          setStatus(status, `${parts.join('')}。`, result.failed ? 'warn' : 'ok');
        }
        announce(status.textContent);
      } catch (err) {
        if (state.me !== me) return;
        reportFailure(status, '没有跑完：', err);
      }
    });
    if (state.me !== me) return;
    loadQueue();
    loadStats();
    if (state.selected) loadDetail(state.selected);
  }

  // ================================================================ 启动

  function init() {
    if (CONFIG.platform_name) $('platform-name').textContent = CONFIG.platform_name;
    $('platform-note').textContent = CONFIG.platform_note || '';
    fillOptions($('filter-state'), [[UNREVIEWED, label(L.nextSteps, 'run_rules')], ...FLOW.map((s) => [s, label(L.states, s)])]);
    fillOptions($('filter-status'), Object.entries(L.statuses));
    fillOptions($('filter-rule'), Object.keys(RULES).map((rule) => [rule, ruleText(rule)]));
    if (LIMITS.reason_code) $('reason-code').maxLength = LIMITS.reason_code;
    if (LIMITS.note) $('reason-text').maxLength = LIMITS.note;

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
    $('stats-form').addEventListener('submit', (event) => event.preventDefault());
    $('stats-range').addEventListener('change', () => {
      if (state.me) loadStats();
    });
    $('queue-filter').addEventListener('submit', (event) => event.preventDefault());
    for (const id of ['filter-state', 'filter-status', 'filter-rule']) {
      $(id).addEventListener('change', () => {
        if (state.me) loadQueue();
      });
    }
    $('run-rules').addEventListener('click', runRules);
    $('queue-refresh').addEventListener('click', () => {
      if (!state.me) return;
      loadQueue();
      loadStats();
    });
    $('confirm-go').addEventListener('click', doConfirm);
    $('confirm-cancel').addEventListener('click', () => {
      if (!confirmCtx.busy) closeConfirm();
    });
    $('confirm-dialog').addEventListener('cancel', (event) => {
      event.preventDefault(); // Esc：执行中不关（结果回来时还要用到 pending），否则正常关闭
      if (!confirmCtx.busy) closeConfirm();
    });

    const saved = storage.get(STORAGE_TOKEN);
    if (saved) signIn(saved);
    else $('token').focus();
  }

  init();
})();
