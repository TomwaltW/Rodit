/*
 * Rodit 客户演示站（单页，按 # 路由）。只调用同源的 /showcase/api/ 接口；服务端和用户给的文字一律用
 * textContent 写进页面（不拼接 HTML 字符串），图表用 DOM / SVG 手画，不引用任何外部资源。
 * 演示站没有登录，不保存任何令牌；sessionStorage 里只记当前平台和选中的会话（方便刷新后接着讲）。
 */
'use strict';

(function () {
  const CONFIG = JSON.parse(document.getElementById('showcase-config').textContent);
  const API = '/showcase/api';
  const ROUTES = ['overview', 'workbench', 'aftersale', 'dashboard', 'compliance'];
  const PLATFORM_KEY = 'rodit.showcase.platform';
  const SCENARIO_KEY = 'rodit.showcase.scenario.';
  // SVG 的命名空间取自页面里已有的 SVG（脚本里不写任何网址）
  const SVG_NS = document.querySelector('.brand-mark').namespaceURI;
  const PLATFORMS = {};
  CONFIG.platforms.forEach((p) => {
    PLATFORMS[p.id] = p;
  });

  const state = {
    platform: initialPlatform(),
    route: 'overview',
    loaded: {}, // 平台 → 是否已经组装过（第一次要等回放历史）
    scenarios: {}, // 平台 → /scenarios 的结果
    active: {}, // 平台 → 选中的场景 ID
    aftersales: {}, // 平台 → /aftersales 的结果
    selected: {}, // 平台 → 选中的售后单
    results: {}, // 平台:售后单 → 最近一次确认的结果
    traceOpen: false,
    busy: false,
  };

  // ================================================================ 工具

  function $(id) {
    return document.getElementById(id);
  }

  function storageGet(key) {
    try {
      return window.sessionStorage.getItem(key);
    } catch (err) {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      window.sessionStorage.setItem(key, value);
    } catch (err) {
      /* 无痕模式等：不记也能用 */
    }
  }

  function initialPlatform() {
    const saved = storageGet(PLATFORM_KEY);
    return saved && PLATFORMS[saved] ? saved : CONFIG.default_platform;
  }

  /** 建一个元素：opts.cls / opts.text / opts.attrs / opts.on；children 是节点或字符串。 */
  function el(tag, opts, children) {
    const node = document.createElement(tag);
    const o = opts || {};
    if (o.cls) node.className = o.cls;
    if (o.text !== undefined && o.text !== null) node.textContent = String(o.text);
    if (o.attrs) {
      Object.keys(o.attrs).forEach((k) => {
        const v = o.attrs[k];
        if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : String(v));
      });
    }
    if (o.on) {
      Object.keys(o.on).forEach((k) => node.addEventListener(k, o.on[k]));
    }
    (children || []).forEach((c) => {
      if (c === null || c === undefined || c === false) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function svgEl(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.keys(attrs || {}).forEach((k) => node.setAttribute(k, String(attrs[k])));
    (children || []).forEach((c) => {
      if (!c) return;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  function pct(v, digits) {
    if (v === null || v === undefined) return '—';
    return (v * 100).toFixed(digits === undefined ? 1 : digits) + '%';
  }

  function ms(v) {
    if (v === null || v === undefined) return '—';
    if (v >= 1000) return (v / 1000).toFixed(2) + ' 秒';
    return Math.round(v) + ' ms';
  }

  function num(v) {
    if (v === null || v === undefined) return '—';
    return Number(v).toLocaleString('zh-CN');
  }

  function metricValue(v, unit) {
    if (unit === 'ratio') return pct(v, 1);
    if (unit === 'seconds') return Math.round(v) + ' 秒';
    return v.toFixed(1) + ' 小时';
  }

  /** 服务端给的时间都是北京时间（+08:00）：直接取字面的月日时分，不随浏览器时区变。 */
  function shortTime(iso) {
    const m = /^\d{4}-(\d\d)-(\d\d)T(\d\d):(\d\d)/.exec(iso || '');
    return m ? m[1] + '-' + m[2] + ' ' + m[3] + ':' + m[4] : '';
  }

  function width(node, fraction) {
    node.style.width = Math.max(0, Math.min(1, fraction)) * 100 + '%';
  }

  function left(node, fraction) {
    node.style.left = Math.max(0, Math.min(1, fraction)) * 100 + '%';
  }

  // ================================================================ 接口

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, options) {
    const o = options || {};
    const init = { method: o.method || 'GET', headers: { Accept: 'application/json' }, credentials: 'same-origin' };
    if (o.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(o.body);
    }
    let response;
    try {
      response = await fetch(API + path, init);
    } catch (err) {
      throw new ApiError(0, '连不上演示服务，请确认 python -m rodit.showcase 还在运行');
    }
    let data = null;
    try {
      data = await response.json();
    } catch (err) {
      data = null;
    }
    if (!response.ok) {
      const message = (data && data.message) || '请求失败（HTTP ' + response.status + '）';
      throw new ApiError(response.status, message);
    }
    return data;
  }

  function q(platform) {
    return '?platform=' + encodeURIComponent(platform);
  }

  async function withLoading(platform, task) {
    const loading = $('loading');
    if (!state.loaded[platform]) {
      loading.textContent = '正在准备' + PLATFORMS[platform].name + '演示单元：组装问答引擎，回放近 5 周的合成数据……';
      loading.hidden = false;
    }
    try {
      const result = await task();
      state.loaded[platform] = true;
      return result;
    } finally {
      loading.hidden = true;
    }
  }

  let toastTimer = null;
  function toast(message, isError) {
    const node = $('toast');
    node.textContent = message;
    node.classList.toggle('is-error', Boolean(isError));
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      node.hidden = true;
    }, isError ? 6000 : 3800);
  }

  function fail(err) {
    toast(err && err.message ? err.message : String(err), true);
  }

  // ================================================================ 浮动提示（图表）

  const tip = $('chart-tip');

  function attachTip(node, content) {
    function showAt(x, y) {
      const lines = content();
      clear(tip);
      tip.appendChild(el('strong', { text: lines[0] }));
      lines.slice(1).forEach((line) => tip.appendChild(el('span', { text: line })));
      tip.hidden = false;
      const rect = tip.getBoundingClientRect();
      const maxX = window.innerWidth - rect.width - 8;
      tip.style.left = Math.max(8, Math.min(maxX, x + 12)) + 'px';
      tip.style.top = Math.max(8, y - rect.height - 12) + 'px';
    }
    node.addEventListener('pointermove', (e) => showAt(e.clientX, e.clientY));
    node.addEventListener('pointerleave', () => {
      tip.hidden = true;
    });
    node.addEventListener('focus', () => {
      const r = node.getBoundingClientRect();
      showAt(r.left + r.width / 2, r.top);
    });
    node.addEventListener('blur', () => {
      tip.hidden = true;
    });
  }

  // ================================================================ 确认弹窗

  const dialog = $('confirm-dialog');

  /** 弹出确认框；返回 Promise：确认时是 {extra}（额外表单的值），取消时是 null。 */
  function confirmDialog(opts) {
    $('dialog-title').textContent = opts.title;
    $('dialog-lead').textContent = opts.lead;
    const details = clear($('dialog-details'));
    (opts.details || []).forEach((pair) => {
      details.appendChild(el('dt', { text: pair[0] }));
      details.appendChild(el('dd', { text: pair[1] }));
    });
    const extra = clear($('dialog-extra'));
    let select = null;
    if (opts.select) {
      const id = 'dialog-select';
      extra.appendChild(el('label', { cls: 'field-label', text: opts.select.label, attrs: { for: id } }));
      select = el('select', { attrs: { id } });
      Object.keys(opts.select.options).forEach((k) => select.appendChild(el('option', { text: opts.select.options[k], attrs: { value: k } })));
      extra.appendChild(select);
    }
    $('dialog-ok').textContent = opts.ok || '确认执行';
    return new Promise((resolve) => {
      function done(value) {
        $('dialog-ok').removeEventListener('click', onOk);
        $('dialog-cancel').removeEventListener('click', onCancel);
        dialog.removeEventListener('cancel', onCancel);
        if (dialog.open) dialog.close();
        resolve(value);
      }
      function onOk() {
        done({ extra: select ? select.value : null });
      }
      function onCancel(e) {
        if (e) e.preventDefault();
        done(null);
      }
      $('dialog-ok').addEventListener('click', onOk);
      $('dialog-cancel').addEventListener('click', onCancel);
      dialog.addEventListener('cancel', onCancel);
      dialog.showModal();
      $('dialog-cancel').focus();
    });
  }

  $('dialog-form').addEventListener('submit', (e) => e.preventDefault());

  // ================================================================ 平台与路由

  function renderPlatformSwitch() {
    const box = clear($('platform-switch'));
    CONFIG.platforms.forEach((p) => {
      const btn = el(
        'button',
        {
          cls: 'platform-btn',
          attrs: { type: 'button', 'data-platform': p.id, 'aria-pressed': p.id === state.platform ? 'true' : 'false' },
          on: { click: () => setPlatform(p.id) },
        },
        [
          el('span', { cls: 'pdot pdot-' + p.id, attrs: { 'aria-hidden': 'true' } }),
          el('span', { cls: 'pname-full', text: p.name }),
          el('span', { cls: 'pname-short', text: p.short_name, attrs: { 'aria-hidden': 'true' } }),
        ]
      );
      box.appendChild(btn);
    });
  }

  function applyPlatform() {
    document.body.dataset.platform = state.platform;
    document.querySelectorAll('.platform-btn').forEach((b) => {
      b.setAttribute('aria-pressed', b.dataset.platform === state.platform ? 'true' : 'false');
    });
    document.querySelectorAll('[data-bind="platform-name"]').forEach((n) => {
      n.textContent = PLATFORMS[state.platform].name;
    });
  }

  function setPlatform(platform) {
    if (platform === state.platform) return;
    state.platform = platform;
    storageSet(PLATFORM_KEY, platform);
    applyPlatform();
    loadRoute();
  }

  function routeFromHash() {
    const name = (window.location.hash || '').replace(/^#\/?/, '');
    return ROUTES.indexOf(name) >= 0 ? name : 'overview';
  }

  function showRoute(route) {
    state.route = route;
    ROUTES.forEach((r) => {
      $('page-' + r).hidden = r !== route;
    });
    document.querySelectorAll('.main-nav a').forEach((a) => {
      if (a.dataset.route === route) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    const titles = {
      overview: '产品概览',
      workbench: '智能辅助工作台',
      aftersale: '售后审核助手',
      dashboard: '数据看板',
      compliance: '合规与部署',
    };
    document.title = 'Rodit · ' + titles[route];
  }

  function loadRoute() {
    const route = state.route;
    if (route === 'overview') loadOverview();
    else if (route === 'workbench') loadWorkbench();
    else if (route === 'aftersale') loadAftersales();
    else if (route === 'dashboard') loadDashboard();
  }

  function onHashChange() {
    const route = routeFromHash();
    const changed = route !== state.route;
    showRoute(route);
    loadRoute();
    if (changed) {
      window.scrollTo(0, 0);
      $('main').focus({ preventScroll: true });
    }
  }

  // ================================================================ 产品概览

  async function loadOverview() {
    const platform = state.platform;
    try {
      const data = await withLoading(platform, () => api('/overview' + q(platform)));
      if (platform !== state.platform) return;
      renderTargets(data.targets);
    } catch (err) {
      fail(err);
    }
  }

  function targetValue(t) {
    if (t.unit === 'ratio') return pct(t.value, 1);
    if (t.unit === 'ms') return ms(t.value);
    if (t.unit === 'count') return num(t.value);
    if (t.unit === 'pending') return '待抽检';
    return t.value || '—';
  }

  function statusBadge(ok) {
    if (ok === true) return el('span', { cls: 'badge badge-ok', text: '✓ 达标' });
    if (ok === false) return el('span', { cls: 'badge badge-danger', text: '✗ 未达标' });
    return el('span', { cls: 'badge', text: '后续评估' });
  }

  function renderTargets(targets) {
    const box = clear($('targets'));
    targets.forEach((t) => {
      const value = el('p', { cls: 'kpi-value' + (t.unit === 'text' || t.unit === 'pending' ? ' kpi-value-text' : ''), text: targetValue(t) });
      box.appendChild(
        el('article', { cls: 'kpi', attrs: { 'data-key': t.key } }, [
          el('div', { cls: 'kpi-head' }, [el('p', { cls: 'kpi-label', text: t.label }), statusBadge(t.ok)]),
          value,
          el('p', { cls: 'kpi-target', text: '目标 ' + t.target }),
          el('p', { cls: 'kpi-note', text: t.note }),
        ])
      );
    });
  }

  // ================================================================ 工作台

  async function loadWorkbench() {
    const platform = state.platform;
    try {
      const data = await withLoading(platform, () => api('/scenarios' + q(platform)));
      if (platform !== state.platform) return;
      state.scenarios[platform] = data;
      const saved = storageGet(SCENARIO_KEY + platform);
      if (!state.active[platform]) {
        // 只恢复服务端还有记录的会话（服务重启、重置之后不恢复，免得停在"正在发出"）
        const restored = data.scenarios.find((s) => s.id === saved);
        state.active[platform] = restored && restored.bubbles.length ? saved : null;
      } else if (!data.scenarios.some((s) => s.id === state.active[platform] && s.bubbles.length)) {
        state.active[platform] = null;
      }
      renderWorkbench();
    } catch (err) {
      fail(err);
    }
  }

  function scenarioData() {
    const data = state.scenarios[state.platform];
    return data ? data.scenarios : [];
  }

  function activeScenario() {
    const id = state.active[state.platform];
    return scenarioData().find((s) => s.id === id) || null;
  }

  function renderWorkbench() {
    const data = state.scenarios[state.platform];
    if (!data) return;
    $('mode-label').textContent = data.shop_name + ' · ' + data.access_label;
    $('ai-mode').textContent = data.access === 'C' ? '模式 C' : '模式 B';
    $('conv-count').textContent = '（' + data.scenarios.length + '）';
    renderConversations();
    renderChat();
    const s = activeScenario();
    renderAI(s ? s.latest : null);
  }

  function lastText(s) {
    const bubbles = s.bubbles || [];
    return bubbles.length ? bubbles[bubbles.length - 1].text : s.text;
  }

  function renderConversations() {
    const list = clear($('conversations'));
    const active = state.active[state.platform];
    scenarioData().forEach((s) => {
      const btn = el(
        'button',
        {
          cls: 'conv-btn',
          attrs: { type: 'button', 'data-scenario': s.id, 'aria-current': s.id === active ? 'true' : 'false' },
          on: { click: () => selectScenario(s.id) },
        },
        [
          el('span', { cls: 'avatar', text: s.title.slice(0, 1), attrs: { 'aria-hidden': 'true' } }),
          el('span', {}, [
            el('span', { cls: 'conv-top' }, [el('span', { cls: 'conv-title', text: s.title }), el('span', { cls: 'conv-section', text: s.section })]),
            el('span', { cls: 'conv-preview', text: lastText(s) }),
            el('span', { cls: 'conv-tags' }, [
              el('span', { cls: 'badge', text: s.intent_label }),
              s.latest ? el('span', { cls: 'badge badge-ok', text: '已处理' }) : null,
            ]),
          ]),
        ]
      );
      list.appendChild(el('li', {}, [btn]));
    });
  }

  function contextCard(card) {
    const node = clear($('chat-card'));
    if (!card) {
      node.hidden = true;
      return;
    }
    node.hidden = false;
    const isOrder = card.kind === 'order';
    const meta = isOrder
      ? ['订单 ' + card.id, card.status, card.logistics ? '物流：' + card.logistics : null, '¥' + card.amount].filter(Boolean).join(' · ')
      : ['商品 ' + card.id, card.price ? '¥' + card.price : null, card.category].filter(Boolean).join(' · ');
    node.appendChild(el('span', { cls: 'context-thumb', text: isOrder ? '单' : '品', attrs: { 'aria-hidden': 'true' } }));
    node.appendChild(
      el('span', {}, [
        el('span', { cls: 'context-title', text: (isOrder ? '订单卡片：' : '商品卡片：') + card.title }),
        el('span', { cls: 'context-meta', text: meta }),
      ])
    );
  }

  function renderChat() {
    const s = activeScenario();
    const list = clear($('messages'));
    const composer = $('composer-text');
    const send = $('composer-send');
    if (!s) {
      $('chat-title').textContent = '选择左边的一位买家';
      $('chat-section').textContent = '每个会话是方案 §11 的一个核心流程，点开就会以买家身份发出预设问题。';
      $('chat-expect').hidden = true;
      contextCard(null);
      composer.disabled = true;
      send.disabled = true;
      list.appendChild(el('li', { cls: 'msg msg-system', text: '还没有选中会话' }));
      return;
    }
    $('chat-title').textContent = '买家 · ' + s.title;
    $('chat-section').textContent = '方案 ' + s.section + (s.note ? ' · ' + s.note : '');
    const expect = $('chat-expect');
    expect.textContent = '会看到：' + s.expect;
    expect.hidden = false;
    contextCard(s.card);
    composer.disabled = state.busy;
    send.disabled = state.busy;
    if (!s.bubbles.length) {
      list.appendChild(el('li', { cls: 'msg msg-system', text: '正在以买家身份发出：' + s.text }));
    }
    s.bubbles.forEach((b) => {
      const isBuyer = b.role === 'buyer';
      const who = isBuyer ? '买家' : '客服（' + (b.edited ? '编辑后发送' : '采纳推荐') + '）';
      list.appendChild(
        el('li', { cls: 'msg ' + (isBuyer ? 'msg-buyer' : 'msg-agent') }, [
          el('span', { cls: 'msg-who', text: who + ' · ' + shortTime(b.at) }),
          el('span', { cls: 'bubble', text: b.text }),
        ])
      );
    });
    list.scrollTop = list.scrollHeight;
  }

  function selectScenario(id) {
    if (state.busy) return; // 上一条还在处理：先等它出结果
    state.active[state.platform] = id;
    storageSet(SCENARIO_KEY + state.platform, id);
    renderConversations();
    renderChat();
    const s = activeScenario();
    renderAI(s ? s.latest : null);
    if (s && !s.bubbles.length) sendBuyer(null);
  }

  async function sendBuyer(text) {
    const platform = state.platform;
    const s = activeScenario();
    if (!s || state.busy) return;
    state.busy = true;
    $('composer-send').disabled = true;
    $('composer-text').disabled = true;
    const box = clear($('suggestions'));
    box.appendChild(el('p', { cls: 'empty', text: '问答引擎处理中：懂 → 找 → 写 → 判……' }));
    try {
      const body = { platform, scenario: s.id };
      if (text) body.text = text;
      const view = await api('/chat', { method: 'POST', body });
      s.bubbles.push(view.message);
      s.latest = view;
      if (text) $('composer-text').value = '';
    } catch (err) {
      fail(err);
    } finally {
      state.busy = false;
    }
    if (platform !== state.platform) return;
    renderConversations();
    renderChat();
    renderAI(activeScenario() ? activeScenario().latest : null);
    $('composer-text').disabled = false;
    $('composer-send').disabled = false;
  }

  $('composer').addEventListener('submit', (e) => {
    e.preventDefault();
    const text = $('composer-text').value.trim();
    if (!text) {
      toast('先以买家身份输入一句话', true);
      return;
    }
    sendBuyer(text);
  });

  $('composer-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      $('composer').requestSubmit();
    }
  });

  function renderAI(view) {
    const alerts = clear($('ai-alerts'));
    const box = clear($('suggestions'));
    const simulated = $('ai-simulated');
    const toggle = $('trace-toggle');
    if (!view) {
      simulated.hidden = true;
      box.appendChild(el('p', { cls: 'empty', text: '买家发来消息后，推荐回答出现在这里。' }));
      toggle.disabled = true;
      setTrace(false);
      return;
    }
    const result = view.result;
    result.alerts.forEach((a) => alerts.appendChild(el('p', { cls: 'alert', text: a })));
    const note = view.trace && view.trace.simulated_note;
    simulated.hidden = !note;
    simulated.textContent = note ? note + '（§11.5）：第一稿被质检拦下，下面是重写后通过的回答' : '';
    if (result.suggestions.length) {
      result.suggestions.forEach((card) => box.appendChild(suggestionCard(card)));
    } else {
      const parts = [el('strong', { text: '没有推荐' }), el('p', { text: result.no_suggestion_reason || '客服自行处理' })];
      if (result.missing_knowledge.length) {
        parts.push(el('p', { cls: 'small muted', text: '缺少的知识（已提醒商家补充）：' }));
        parts.push(el('ul', {}, result.missing_knowledge.map((m) => el('li', { text: m.label }))));
      }
      box.appendChild(el('div', { cls: 'no-suggestion' }, parts));
    }
    if (result.suggestions.length && result.missing_knowledge.length) {
      box.appendChild(el('p', { cls: 'small muted', text: '提醒商家补知识：' + result.missing_knowledge.map((m) => m.label).join('、') }));
    }
    toggle.disabled = false;
    renderTrace(view);
    setTrace(state.traceOpen);
  }

  function setTrace(open) {
    const toggle = $('trace-toggle');
    const panel = $('trace');
    panel.hidden = !open || toggle.disabled;
    toggle.setAttribute('aria-expanded', open && !toggle.disabled ? 'true' : 'false');
    toggle.textContent = open && !toggle.disabled ? '收起 AI 的思考过程' : 'AI 是怎么想的';
  }

  $('trace-toggle').addEventListener('click', () => {
    state.traceOpen = !state.traceOpen;
    setTrace(state.traceOpen);
    if (state.traceOpen) {
      const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      $('trace').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    }
  });

  const KIND_CLS = { faq: 'badge-ok', template: 'badge-kind', llm: 'badge-accent' };

  function suggestionCard(card) {
    const data = state.scenarios[state.platform];
    const modeC = data && data.access === 'C';
    const node = el('article', { cls: 'sg-card', attrs: { 'data-status': card.status, 'data-suggestion': card.suggestion_id } });
    const textId = 'sg-text-' + card.suggestion_id;
    node.appendChild(
      el('div', { cls: 'sg-head' }, [
        el('span', { cls: 'badge ' + (KIND_CLS[card.kind] || ''), text: card.kind_label }),
        el('span', { cls: 'badge ' + (card.confidence === 'high' ? 'badge-ok' : 'badge-warn'), text: '置信度 ' + card.confidence_label }),
        el('span', { cls: 'sg-intent', text: card.intent_label }),
      ])
    );
    const textarea = el('textarea', {
      cls: 'sg-text',
      attrs: { id: textId, rows: 3, maxlength: CONFIG.max_text_chars, 'aria-label': '推荐回答（可以编辑）' },
    });
    textarea.value = card.text;
    node.appendChild(textarea);
    if (card.evidence.length) {
      node.appendChild(el('div', { cls: 'chips', attrs: { 'aria-label': '依据' } }, card.evidence.map((e) => el('span', { cls: 'chip', text: e.label }))));
    } else {
      node.appendChild(el('p', { cls: 'small muted', text: '依据：话术模板' }));
    }
    card.risk_hints.forEach((h) => node.appendChild(el('p', { cls: 'risk', text: h })));
    if (card.status !== 'shown') {
      textarea.readOnly = true;
      const labels = { sent: '✓ 已原样发送', edited: '✓ 已编辑后发送', rejected: '已标记不合适', expired: '已过期（买家又发了新消息）' };
      node.appendChild(el('p', { cls: 'sg-done' + (card.status === 'sent' || card.status === 'edited' ? '' : ' sg-done-muted'), text: labels[card.status] || card.status }));
      return node;
    }
    const actions = el('div', { cls: 'sg-actions' });
    const sendLabel = modeC ? '复制并标记已发送' : '发送';
    actions.appendChild(
      el('button', {
        cls: 'btn btn-primary btn-small',
        text: sendLabel,
        attrs: { type: 'button', 'data-action': 'sent' },
        on: {
          click: () => {
            const text = textarea.value.trim();
            if (text !== card.text.trim()) sendFeedback(card, 'edited', { text });
            else sendFeedback(card, 'sent', {});
          },
        },
      })
    );
    actions.appendChild(
      el('button', {
        cls: 'btn btn-secondary btn-small',
        text: '编辑后发送',
        attrs: { type: 'button', 'data-action': 'edited' },
        on: {
          click: () => {
            const text = textarea.value.trim();
            if (text === card.text.trim()) {
              textarea.focus();
              textarea.setSelectionRange(textarea.value.length, textarea.value.length);
              toast('改好推荐回答后，再点「编辑后发送」');
              return;
            }
            sendFeedback(card, 'edited', { text });
          },
        },
      })
    );
    const menu = el('div', { cls: 'reject-menu', attrs: { hidden: true } }, [el('p', { text: '为什么不合适？（§5.8 反馈闭环）' })]);
    Object.keys(CONFIG.reject_reasons).forEach((reason) => {
      menu.appendChild(
        el('button', {
          cls: 'btn btn-quiet btn-small',
          text: CONFIG.reject_reasons[reason],
          attrs: { type: 'button', 'data-reason': reason },
          on: { click: () => sendFeedback(card, 'rejected', { reason }) },
        })
      );
    });
    const rejectBtn = el('button', {
      cls: 'btn btn-quiet btn-small',
      text: '不合适',
      attrs: { type: 'button', 'data-action': 'rejected', 'aria-expanded': 'false' },
      on: {
        click: () => {
          menu.hidden = !menu.hidden;
          rejectBtn.setAttribute('aria-expanded', menu.hidden ? 'false' : 'true');
        },
      },
    });
    actions.appendChild(rejectBtn);
    node.appendChild(actions);
    node.appendChild(menu);
    return node;
  }

  async function sendFeedback(card, action, extra) {
    const platform = state.platform;
    const s = activeScenario();
    try {
      const body = { platform, suggestion_id: card.suggestion_id, action };
      if (extra.text) body.text = extra.text;
      if (extra.reason) body.reason = extra.reason;
      if (action !== 'rejected' && state.scenarios[platform].access === 'C' && navigator.clipboard) {
        navigator.clipboard.writeText(extra.text || card.text).catch(() => {});
      }
      const out = await api('/feedback', { method: 'POST', body });
      card.status = action;
      if (out.bubble && s) s.bubbles.push(out.bubble);
      let message = '已记录：' + out.action_label;
      if (action === 'sent') message += ' · 计入采纳率';
      if (out.candidate) message += ' · 生成候选知识，等商家审核后进 FAQ';
      else if (action === 'edited' && out.candidate_skipped) message += ' · ' + out.candidate_skipped;
      if (out.reason_label) message += '（' + out.reason_label + '）';
      if (out.gap) message += ' · 提醒商家补知识：' + out.gap;
      if (out.labeled && action === 'rejected') message += ' · 进判断层标注集';
      toast(message);
    } catch (err) {
      fail(err);
    }
    if (platform !== state.platform) return;
    renderConversations();
    renderChat();
    const active = activeScenario();
    renderAI(active ? active.latest : null);
  }

  // ---------------------------------------------------------------- 过程面板

  function bar(fraction, opts) {
    const o = opts || {};
    const track = el('span', { cls: 'track' });
    const fill = el('span', { cls: 'track-fill' + (o.cls ? ' ' + o.cls : '') });
    width(fill, fraction);
    track.appendChild(fill);
    if (o.mark !== undefined && o.mark !== null) {
      const mark = el('span', { cls: 'track-mark', attrs: { title: o.markLabel || '' } });
      left(mark, o.mark);
      track.appendChild(mark);
    }
    return track;
  }

  function traceStep(badge, title, sub, children) {
    return el('li', { cls: 'trace-step' }, [
      el('span', { cls: 'trace-badge', text: badge, attrs: { 'aria-hidden': 'true' } }),
      el('div', {}, [el('h3', {}, [title, sub ? el('span', { cls: 'muted', text: sub }) : null])].concat(children)),
    ]);
  }

  function understandStep(u) {
    if (!u) return traceStep('懂', '懂问题', '', [el('p', { cls: 'muted', text: '这条消息没有走到路由（店铺停用或系统降级）。' })]);
    const probs = el('div', { cls: 'trace-box' }, [el('h4', { text: '问题类型（choice，带概率）' })]);
    u.top_intents.forEach((t) => {
      probs.appendChild(
        el('div', { cls: 'prob-row' }, [
          el('span', { cls: 'row-label', text: t.label }),
          bar(t.p, { mark: u.assist_threshold, markLabel: '智能辅助阈值' }),
          el('span', { cls: 'row-value', text: pct(t.p, 0) }),
        ])
      );
    });
    if (u.parts.length) {
      probs.appendChild(el('p', { cls: 'small muted', text: '分条：' + u.parts.map((p) => p.label + ' ' + pct(p.p, 0)).join('、') }));
    }
    const gauges = el('div', { cls: 'trace-box' }, [el('h4', { text: '其他判断（noul / score，竖线是阈值）' })]);
    const rows = [
      ['需要人工', u.need_human, 1, u.need_human_threshold, pct(u.need_human, 0)],
      ['情绪 · ' + u.emotion_label, u.emotion, 4, u.emotion_threshold, u.emotion.toFixed(1) + ' / 4'],
      ['多个问题', u.multi_intent, 1, u.multi_intent_threshold, pct(u.multi_intent, 0)],
    ];
    rows.forEach((r) => {
      const over = r[1] >= r[3];
      gauges.appendChild(
        el('div', { cls: 'gauge-row' }, [
          el('span', { cls: 'row-label', text: r[0] }),
          bar(r[1] / r[2], { mark: r[3] / r[2], cls: over ? 'is-bad' : '' }),
          el('span', { cls: 'row-value', text: r[4] }),
        ])
      );
    });
    const rule = el('div', { cls: 'rule-line' }, [
      el('span', { cls: 'badge badge-accent', text: '§6.6 规则 ' + u.rule }),
      el('span', { text: u.rule_text }),
    ]);
    const provider = el('p', {
      cls: 'small muted',
      text: '判断层：' + u.provider_label + (u.questions_ver ? '（题目版本 ' + u.questions_ver + '）' : '') + (u.degraded ? ' · 已降级为关键词规则' : ''),
    });
    const head = u.intent_label + ' · 置信度 ' + pct(u.confidence, 0) + (u.high_risk ? ' · 高风险类型' : '');
    return traceStep('懂', '懂问题', head, [el('div', { cls: 'trace-grid' }, [probs, gauges]), rule, provider]);
  }

  function findStep(parts) {
    const children = [];
    parts.forEach((part) => {
      const f = part.find;
      if (parts.length > 1) children.push(el('p', { cls: 'small', text: '分条 · ' + part.intent_label }));
      if (!f) {
        children.push(el('p', { cls: 'muted small', text: '这一条没有检索（话术模板或不需要依据）。' }));
        return;
      }
      const list = el('ul', { cls: 'ev-list' });
      f.evidence.forEach((e) => {
        list.appendChild(
          el('li', { cls: 'ev' + (e.cited ? ' ev-cited' : '') }, [
            el('span', { cls: 'ev-label', text: e.label }),
            el('span', { cls: 'ev-meta' }, [
              el('span', { cls: 'badge', text: '优先级 ' + e.priority + ' · ' + e.source_label }),
              e.as_of ? el('span', { cls: 'badge badge-kind', text: e.as_of }) : null,
              e.cited ? el('span', { cls: 'badge badge-ok', text: '✓ 回答引用' }) : null,
            ]),
            el('span', { cls: 'ev-content', text: e.content }),
          ])
        );
      });
      f.overridden.forEach((o) => {
        list.appendChild(
          el('li', { cls: 'ev ev-over' }, [
            el('span', { cls: 'ev-label', text: o.label }),
            el('span', { cls: 'ev-meta' }, [el('span', { cls: 'badge badge-warn', text: '被高优先级覆盖（§5.2.1）' })]),
            el('span', { cls: 'ev-content', text: o.content || o.id }),
          ])
        );
      });
      if (!f.evidence.length && !f.overridden.length) list.appendChild(el('li', { cls: 'empty', text: '没有找到依据' }));
      children.push(list);
      f.missing.forEach((m) => children.push(el('p', { cls: 'small risk', text: '缺少：' + m.label })));
      f.unavailable.forEach((m) => children.push(el('p', { cls: 'small risk', text: '暂时查不到：' + m.label })));
      if (f.clarify) children.push(el('p', { cls: 'small', text: '需要先澄清：' + f.clarify }));
      f.alerts.forEach((a) => children.push(el('p', { cls: 'small risk', text: '提醒客服：' + a })));
    });
    if (!parts.length) children.push(el('p', { cls: 'muted small', text: '这条消息不需要找依据（不推荐、只提醒人工）。' }));
    return traceStep('找', '找依据', '知识中心 + 平台只读工具；冲突时高优先级覆盖低优先级', children);
  }

  function writeStep(parts, note) {
    const children = [];
    parts.forEach((part) => {
      if (parts.length > 1) children.push(el('p', { cls: 'small', text: '分条 · ' + part.intent_label }));
      if (!part.write.length) children.push(el('p', { cls: 'muted small', text: '没有生成。' }));
      part.write.forEach((a) => {
        a.drafts.forEach((d) => {
          const head = [el('span', { cls: 'badge', text: '第 ' + a.attempt + ' 稿' }), el('span', { cls: 'badge ' + (KIND_CLS[d.kind] || ''), text: d.kind_label })];
          if (note && a.attempt === 1 && d.kind === 'llm') head.push(el('span', { cls: 'badge badge-warn', text: note }));
          children.push(
            el('div', { cls: 'draft' }, [
              el('div', { cls: 'draft-head' }, head),
              el('p', { cls: 'draft-text', text: d.text }),
              a.retry_feedback.length ? el('p', { cls: 'draft-retry', text: '带着质检原因重写：' + a.retry_feedback.join('；') }) : null,
            ])
          );
        });
      });
    });
    return traceStep('写', '写回答', 'FAQ / 模板能答的不调大模型；大模型只根据依据写（§5.4）', children);
  }

  function judgeStep(parts) {
    const children = [];
    parts.forEach((part) => {
      if (parts.length > 1 && part.judge.length) children.push(el('p', { cls: 'small', text: '分条 · ' + part.intent_label }));
      part.judge.forEach((c) => {
        const node = el('div', { cls: 'check ' + (c.passed ? 'check-pass' : 'check-fail') }, [
          el('div', { cls: 'check-head' }, [
            el('span', { cls: 'badge ' + (c.passed ? 'badge-ok' : 'badge-danger'), text: c.passed ? '✓ 通过' : '✗ 拦截' }),
            el('span', { cls: 'badge', text: '第 ' + c.attempt + ' 稿 · ' + c.kind_label }),
            el('span', { cls: 'small muted', text: c.model_checked ? '硬规则 + ' + c.provider_label : '只跑硬规则（FAQ / 模板）' }),
          ]),
          el('p', { cls: 'check-text', text: c.text }),
        ]);
        const hits = el('ul', { cls: 'check-list' });
        c.hard_hits.forEach((h) => hits.appendChild(el('li', { cls: h.blocking ? '' : 'ok', text: h.label + '：' + h.detail + (h.blocking ? '' : '（自动修正）') })));
        if (!c.hard_hits.length) hits.appendChild(el('li', { cls: 'ok', text: '硬规则：站外引流、绝对化用语、禁用词、数字一致性都没有命中' }));
        node.appendChild(hits);
        c.scores.forEach((sc) => {
          const limit = sc.min !== null && sc.min !== undefined ? '≥ ' + sc.min : sc.max !== null && sc.max !== undefined ? '≤ ' + sc.max : '';
          const thresholdValue = sc.min !== null && sc.min !== undefined ? sc.min : sc.max;
          node.appendChild(
            el('div', { cls: 'score-row' }, [
              el('span', { cls: 'row-label', text: sc.label }),
              bar(sc.value / sc.scale, {
                mark: thresholdValue !== null && thresholdValue !== undefined ? thresholdValue / sc.scale : null,
                cls: sc.ok ? '' : 'is-bad',
              }),
              el('span', { cls: 'row-value' }, [
                el('span', { cls: sc.ok ? 'ok-mark' : 'bad-mark', text: sc.ok ? '✓ ' : '✗ ' }),
                (sc.scale === 4 ? sc.value.toFixed(1) : sc.value.toFixed(2)) + (limit ? ' ' + limit : ''),
              ]),
            ])
          );
        });
        children.push(node);
      });
    });
    if (!children.length) children.push(el('p', { cls: 'muted small', text: '没有要质检的回答。' }));
    return traceStep('判', '把关', '硬规则 + 判断层质检；任何一条不过就带原因重写一次（§5.5）', children);
  }

  function fallbackStep(fb) {
    const children = [];
    fb.alerts.forEach((a) => children.push(el('p', { cls: 'alert', text: a })));
    if (fb.transfer_to_human) children.push(el('p', { cls: 'alert', text: '已请平台转人工' }));
    if (fb.no_suggestion_reason) children.push(el('p', { text: '不推荐：' + fb.no_suggestion_reason }));
    if (fb.missing_knowledge.length) children.push(el('p', { cls: 'small', text: '提醒商家补知识：' + fb.missing_knowledge.map((m) => m.label).join('、') }));
    if (fb.degraded.length) children.push(el('p', { cls: 'small risk', text: '降级：' + fb.degraded.map((d) => d.label).join('、') }));
    if (!children.length) children.push(el('p', { cls: 'muted', text: '这条不需要兜底：推荐交给客服确认后发送（人在回路，§4.1 原则 3）。' }));
    return traceStep('兜', '兜底', '拿不准的交给人（§5.7）', children);
  }

  function latencyStep(lat) {
    const total = Math.max(1, lat.stages.reduce((sum, s) => sum + s.ms, 0));
    const stack = el('div', { cls: 'latency-stack', attrs: { role: 'img', 'aria-label': '各环节耗时占比' } });
    const legend = el('div', { cls: 'latency-legend' });
    lat.stages.forEach((s) => {
      const seg = el('span', { cls: 'latency-seg', attrs: { tabindex: 0, 'aria-label': s.label + ' ' + s.ms + ' 毫秒' } });
      seg.style.flex = String(Math.max(s.ms, total * 0.01));
      attachTip(seg, () => [s.ms + ' ms', s.label]);
      stack.appendChild(seg);
      legend.appendChild(el('span', { text: s.label + ' ' + s.ms + ' ms' }));
    });
    const e2e = lat.since_received_ms;
    const meter = el('div', { cls: 'gauge-row' }, [
      el('span', { cls: 'row-label', text: '端到端 vs 5 秒' }),
      bar(e2e / lat.target_ms, { mark: 1, cls: lat.within_target ? 'is-good' : 'is-bad' }),
      el('span', { cls: 'row-value', text: ms(e2e) }),
    ]);
    const sub = (lat.within_target ? '✓ ' : '✗ ') + '从收到消息到出推荐 ' + ms(e2e) + '，目标 P95 ≤ 5 秒（§12.1）';
    const note = CONFIG.offline ? el('p', { cls: 'small muted', text: '离线替身不联网，耗时是毫秒级；接真实大模型后路由 ≤ 1 秒、生成 ≤ 2.5 秒、质检 ≤ 1 秒。' }) : null;
    return traceStep('时', '耗时', sub, [stack, legend, el('div', { cls: 'trace-box' }, [meter]), note]);
  }

  function renderTrace(view) {
    const t = view.trace;
    $('trace-total').textContent = '买家：' + view.message.text + ' · 端到端 ' + ms(t.latency.since_received_ms);
    const body = clear($('trace-body'));
    const steps = el('ol', { cls: 'trace-steps' }, [
      understandStep(t.understand),
      findStep(t.parts),
      writeStep(t.parts, t.simulated_note),
      judgeStep(t.parts),
      fallbackStep(t.fallback),
      latencyStep(t.latency),
    ]);
    body.appendChild(steps);
  }

  // ================================================================ 售后审核

  const STATE_LABELS = {};
  CONFIG.review_states.forEach((s) => {
    STATE_LABELS[s.state] = s.label;
  });

  async function loadAftersales() {
    const platform = state.platform;
    try {
      const data = await withLoading(platform, () => api('/aftersales' + q(platform)));
      if (platform !== state.platform) return;
      state.aftersales[platform] = data;
      renderAftersales();
    } catch (err) {
      fail(err);
    }
  }

  function selectedItem() {
    const data = state.aftersales[state.platform];
    if (!data) return null;
    const id = state.selected[state.platform];
    return data.items.find((i) => i.aftersale_id === id) || data.items[0] || null;
  }

  function renderStrip(item) {
    const strip = clear($('state-strip'));
    const visited = new Set();
    if (item) item.review.history.forEach((h) => {
      visited.add(h.from_state);
      visited.add(h.to_state);
    });
    CONFIG.review_states.forEach((s) => {
      const current = item && item.review.state === s.state;
      const cls = 'state-pill' + (current ? ' is-current' : visited.has(s.state) ? ' is-visited' : '');
      strip.appendChild(el('li', {}, [el('span', { cls, text: s.label, attrs: { 'aria-current': current ? 'step' : null, 'data-state': s.state } })]));
    });
    $('strip-caption').textContent = item ? '审核状态机（§7.3.2）· 售后单 ' + item.aftersale_id + ' 当前：' + item.review.state_label : '审核状态机（§7.3.2）';
  }

  function platformNote(data) {
    const notes = {
      taobao: '淘宝：同意退款要子账号权限和短信验证；拒绝前部分场景要先协商，拒绝要非常保守。',
      douyin: '抖店：体验分考核「售后平均审核时长」；拒绝售后必须传原因码。',
      pdd: '拼多多：售后接口在多多云内调用；没有查到拒绝接口，只能同意退款 / 同意退货（§7.1）。',
    };
    return notes[state.platform] + ' 确认人：店长 ' + data.operator + '（合成账号）。';
  }

  function renderAftersales() {
    const data = state.aftersales[state.platform];
    if (!data) return;
    $('aftersale-note').textContent = platformNote(data);
    const box = clear($('aftersales'));
    const current = selectedItem();
    renderStrip(current);
    if (!data.items.length) {
      box.appendChild(el('p', { cls: 'empty', text: '最近 7 天没有售后单' }));
      return;
    }
    data.items.forEach((item) => box.appendChild(aftersaleCard(item, current && current.aftersale_id === item.aftersale_id)));
  }

  function aftersaleCard(item, selected) {
    const r = item.review;
    const card = el('article', {
      cls: 'as-card' + (selected ? ' is-selected' : ''),
      attrs: { 'data-id': item.aftersale_id, 'aria-label': '售后单 ' + item.aftersale_id },
      on: {
        click: (e) => {
          if (e.target.closest('button')) return;
          state.selected[state.platform] = item.aftersale_id;
          renderAftersales();
        },
      },
    });
    card.appendChild(
      el('div', { cls: 'as-head' }, [
        el('div', {}, [
          el('p', { cls: 'as-id', text: item.aftersale_id + ' · 订单 ' + (item.order_id || '—') }),
          el('p', { cls: 'as-title', text: (item.item_title || '') + ' · ' + item.type_label + ' · ' + item.reason_label }),
        ]),
        el('p', { cls: 'as-amount', text: '¥' + item.amount }),
      ])
    );
    const tags = el('div', { cls: 'as-tags' }, [
      el('span', { cls: 'badge', text: '平台：' + item.platform_status_label }),
      el('span', { cls: 'badge badge-accent', text: '审核：' + r.state_label }),
    ]);
    if (r.requires_manager) tags.appendChild(el('span', { cls: 'badge badge-danger', text: 'R6 须店长确认' }));
    r.risk_flags.forEach((f) => tags.appendChild(el('span', { cls: 'badge badge-danger', text: '⚠ ' + f })));
    if (item.evidence_count) tags.appendChild(el('span', { cls: 'badge', text: '凭证 ' + item.evidence_count + ' 张' }));
    card.appendChild(tags);
    if (item.buyer_note) card.appendChild(el('p', { cls: 'small muted', text: '买家说明（已打码）：' + item.buyer_note }));
    card.appendChild(el('ul', { cls: 'as-rules' }, r.rules.map((rule) => el('li', {}, [el('span', { cls: 'rule-id', text: rule.id }), el('span', { text: rule.text })]))));
    card.appendChild(el('div', { cls: 'as-suggest' }, [el('strong', { text: r.suggestion_label || '—' }), el('span', { text: r.reason || '' })]));
    const key = state.platform + ':' + item.aftersale_id;
    const last = state.results[key];
    if (last) card.appendChild(resultBox(last));
    const actions = el('div', { cls: 'as-actions' });
    item.actions.forEach((a) => {
      actions.appendChild(
        el('button', {
          cls: 'btn btn-small ' + (a.action === 'refuse' ? 'btn-danger' : 'btn-primary'),
          text: '确认' + a.label,
          attrs: { type: 'button', 'data-action': a.action },
          on: { click: () => confirmAction(item, a) },
        })
      );
    });
    if (item.can_negotiate) {
      actions.appendChild(el('button', { cls: 'btn btn-secondary btn-small', text: '先协商', attrs: { type: 'button', 'data-action': 'negotiate' }, on: { click: () => negotiate(item, false) } }));
    }
    if (item.can_finish_negotiation) {
      actions.appendChild(el('button', { cls: 'btn btn-secondary btn-small', text: '协商有结果', attrs: { type: 'button', 'data-action': 'negotiate_done' }, on: { click: () => negotiate(item, true) } }));
    }
    if (last && last.ok && !last.duplicate) {
      actions.appendChild(
        el('button', {
          cls: 'btn btn-quiet btn-small',
          text: '再提交一次（演示幂等）',
          attrs: { type: 'button', 'data-action': 'repeat' },
          on: { click: () => runConfirm(item, last.action, last.version, last.reasonCode) },
        })
      );
    }
    if (!actions.childNodes.length) actions.appendChild(el('span', { cls: 'small muted', text: r.state === 'verify' ? '签收核查中：24 小时后仍未找到再建议同意' : r.state === 'evidence' ? '等买家补充凭证' : '不需要操作' }));
    card.appendChild(actions);
    return card;
  }

  function resultBox(res) {
    const box = el('div', { cls: 'as-result' + (res.ok ? '' : ' is-error'), attrs: { role: 'status' } });
    box.appendChild(el('strong', { text: res.duplicate ? '幂等：同一个确认已经执行过，没有再次调用平台' : res.ok ? '✓ 已执行，平台返回成功' : '✗ 平台没有执行' }));
    const dl = el('dl');
    const rows = [
      ['确认人', '店长 ' + res.operator],
      ['重读状态', res.reread],
      ['幂等键', res.key],
      ['平台返回', res.platform],
    ];
    rows.forEach((row) => {
      dl.appendChild(el('dt', { text: row[0] }));
      dl.appendChild(el('dd', { text: row[1] }));
    });
    box.appendChild(dl);
    return box;
  }

  function actionDetails(item, a) {
    return [
      ['售后单', item.aftersale_id + '（' + item.type_label + ' · ' + item.reason_label + '）'],
      ['金额', '¥' + item.amount],
      ['操作', a.label],
      ['命中规则', item.review.rules.map((r) => r.id).join('、') || '—'],
      ['确认人', '店长 ' + state.aftersales[state.platform].operator + '（合成账号）'],
      ['执行前', '重新读取平台状态；版本和你看到的不一致就不执行'],
      ['幂等键', item.aftersale_id + ':' + a.action + ':' + item.status_version],
    ];
  }

  async function confirmAction(item, a) {
    const data = state.aftersales[state.platform];
    const needsReason = a.action === 'refuse' && state.platform === 'douyin';
    const answer = await confirmDialog({
      title: '确认' + a.label + '？',
      lead: '售后写操作必须人确认（§4.1 原则 3、§7.3.2）：系统只给建议，点确认才会调用平台接口。',
      details: actionDetails(item, a),
      select: needsReason ? { label: '拒绝原因码（抖店必填）', options: data.refuse_reasons } : null,
    });
    if (!answer) return;
    await runConfirm(item, a.action, item.status_version, answer.extra);
  }

  async function runConfirm(item, action, version, reasonCode) {
    const platform = state.platform;
    try {
      const body = { platform, action, expected_status_version: version };
      if (reasonCode) body.reason_code = reasonCode;
      const out = await api('/aftersales/' + encodeURIComponent(item.aftersale_id) + '/confirm', { method: 'POST', body });
      const before = out.reread.before;
      const after = out.reread.after;
      const reread = after
        ? before.label + '（版本 ' + before.version + '）→ ' + after.label + '（版本 ' + after.version + '）'
        : before.label + '（版本 ' + before.version + '），没有再调平台';
      const pr = out.platform_result || {};
      state.results[platform + ':' + item.aftersale_id] = {
        ok: Boolean(pr.ok),
        duplicate: out.duplicate,
        operator: out.operator,
        reread,
        key: out.idempotency_key,
        platform: (pr.ok ? '成功' : '失败') + (pr.platform_code ? '（代码 ' + pr.platform_code + '）' : '') + (pr.message ? ' ' + pr.message : ''),
        action,
        version,
        reasonCode,
      };
      toast(out.duplicate ? '幂等：同一操作没有重复执行' : '已执行：' + out.item.review.state_label);
      if (platform !== state.platform) return;
      const data = state.aftersales[platform];
      data.items = data.items.map((i) => (i.aftersale_id === item.aftersale_id ? out.item : i));
      state.selected[platform] = item.aftersale_id;
      renderAftersales();
    } catch (err) {
      fail(err);
    }
  }

  async function negotiate(item, done) {
    const answer = await confirmDialog({
      title: done ? '协商有结果？' : '先和买家协商？',
      lead: '售后写操作必须人确认（§4.1 原则 3、§7.3.2）；协商不调用平台写接口，只改审核状态并留痕。',
      details: [
        ['售后单', item.aftersale_id],
        ['当前', item.review.state_label],
        ['之后', done ? '回到「待客服确认」，重新给建议' : '进入「协商中」'],
        ['确认人', '店长 ' + state.aftersales[state.platform].operator + '（合成账号）'],
      ],
      ok: done ? '协商有结果' : '开始协商',
    });
    if (!answer) return;
    const platform = state.platform;
    try {
      const out = await api('/aftersales/' + encodeURIComponent(item.aftersale_id) + '/negotiate', { method: 'POST', body: { platform, done } });
      toast(done ? '协商有结果，回到待客服确认' : '已进入协商中');
      if (platform !== state.platform) return;
      const data = state.aftersales[platform];
      data.items = data.items.map((i) => (i.aftersale_id === item.aftersale_id ? out.item : i));
      state.selected[platform] = item.aftersale_id;
      renderAftersales();
    } catch (err) {
      fail(err);
    }
  }

  // ================================================================ 数据看板

  async function loadDashboard() {
    const platform = state.platform;
    try {
      const data = await withLoading(platform, () => api('/dashboard' + q(platform)));
      if (platform !== state.platform) return;
      renderDashboard(data);
    } catch (err) {
      fail(err);
    }
  }

  function kpiTile(label, value, target, note, meter) {
    const tile = el('article', { cls: 'kpi' }, [el('p', { cls: 'kpi-label', text: label }), el('p', { cls: 'kpi-value', text: value })]);
    if (meter) {
      const m = el('div', { cls: 'meter', attrs: { 'aria-hidden': 'true' } });
      const fill = el('span', { cls: 'meter-fill' });
      width(fill, meter.value);
      m.appendChild(fill);
      if (meter.mark !== undefined) {
        const mark = el('span', { cls: 'meter-mark' });
        left(mark, meter.mark);
        m.appendChild(mark);
      }
      tile.appendChild(m);
    }
    if (target) tile.appendChild(el('p', { cls: 'kpi-target', text: target }));
    if (note) tile.appendChild(el('p', { cls: 'kpi-note', text: note }));
    return tile;
  }

  let lastDashboard = null;
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.route === 'dashboard' && lastDashboard && lastDashboard.platform === state.platform) {
        weeklyChart($('chart-weekly'), lastDashboard.weekly, lastDashboard.kpis.adoption_target);
      }
    }, 150);
  });

  function renderDashboard(d) {
    lastDashboard = d;
    $('dashboard-asof').textContent = d.synthetic + ' · 截至 ' + shortTime(d.as_of) + ' · ' + PLATFORMS[d.platform].shop_name;
    const k = d.kpis;
    const kpis = clear($('kpis'));
    kpis.appendChild(kpiTile('推荐采纳率', pct(k.adoption_rate), '目标 ≥ ' + pct(k.adoption_target, 0) + '（§1.3）', '按推荐组，' + num(k.groups) + ' 组', { value: k.adoption_rate || 0, mark: k.adoption_target }));
    kpis.appendChild(kpiTile('原样采纳率', pct(k.as_is_rate), '客服原样发送的比例', '半自动的门槛是 ≥ 90%（§13.4）', { value: k.as_is_rate || 0, mark: 0.9 }));
    kpis.appendChild(kpiTile('处理的买家消息', num(k.messages), '近 5 周 + 演示操作', '待商家审核的候选知识 ' + num(k.pending_candidates) + ' 条'));
    kpis.appendChild(kpiTile('质检拦截率', pct(k.interception_rate), '按消息算（§12.4）', '拦下后带原因重写，或不推荐'));
    kpis.appendChild(kpiTile('推荐出现 P95', ms(k.e2e_p95_ms), '目标 ≤ 5 秒（§12.1）', 'P50 ' + ms(k.e2e_p50_ms), { value: (k.e2e_p95_ms || 0) / k.e2e_target_ms, mark: 1 }));
    kpis.appendChild(kpiTile('发出去的违规', num(k.violations), '目标 0（§1.3）', '提醒商家补的知识 ' + num(k.knowledge_gaps) + ' 处'));
    weeklyChart($('chart-weekly'), d.weekly, k.adoption_target);
    hbars(
      $('chart-reasons'),
      d.reject_reasons.map((r) => ({ label: r.label, value: r.count, text: r.count + ' 次' })),
      { max: Math.max(1, ...d.reject_reasons.map((r) => r.count)), unit: '次' }
    );
    hbars(
      $('chart-intents'),
      d.intents.slice(0, 10).map((r) => ({ label: r.label, value: r.count, text: pct(r.share, 0), tip: r.count + ' 条消息' })),
      { max: Math.max(1, ...d.intents.map((r) => r.count)) }
    );
    hbars(
      $('chart-by-intent'),
      d.by_intent.slice(0, 8).map((r) => ({ label: r.label, value: r.adoption_rate || 0, text: pct(r.adoption_rate, 0), tip: r.groups + ' 组推荐 · 原样 ' + pct(r.as_is_rate, 0) })),
      { max: 1, mark: k.adoption_target, markLabel: '目标 60%' }
    );
    qaPanel($('chart-qa'), d.qa);
    latencyPanel($('chart-latency'), d.latency);
    loopPanel($('chart-loop'), d.loop);
    metricsPanel($('chart-metrics'), d.platform_metrics);
  }

  const LABEL_SOURCES = {
    feedback_adopt: '原样采纳（正样本）',
    feedback_edit: '编辑后发送',
    feedback_reject: '理解错误（纠正问题类型）',
    manual: '人工标注',
  };

  function loopPanel(container, loop) {
    const box = clear(container);
    box.appendChild(
      el('div', { cls: 'stat-row' }, [
        el('div', {}, [el('span', { cls: 'stat-big', text: num(loop.pending_candidates) }), el('span', { cls: 'small muted', text: '条候选知识等商家审核' })]),
        el('div', {}, [el('span', { cls: 'stat-big', text: num(loop.knowledge_gaps) }), el('span', { cls: 'small muted', text: '次提醒商家补知识' })]),
        el('div', {}, [el('span', { cls: 'stat-big', text: num(loop.labels) }), el('span', { cls: 'small muted', text: '条判断层标注样本' })]),
      ])
    );
    const holder = el('div');
    box.appendChild(el('p', { cls: 'small muted', text: '最常缺的知识' }));
    box.appendChild(holder);
    hbars(
      holder,
      loop.top_gaps.map((g) => ({ label: g.label, value: g.count, text: g.count + ' 次' })),
      { max: Math.max(1, ...loop.top_gaps.map((g) => g.count)) }
    );
    const sources = Object.keys(loop.labels_by_source).map((k) => (LABEL_SOURCES[k] || k) + ' ' + loop.labels_by_source[k]);
    if (sources.length) box.appendChild(el('p', { cls: 'small muted', text: '标注来源：' + sources.join('；') + '。攒够 2,000–5,000 条后训练路由分类器（§6.4）。' }));
  }

  function hbars(container, rows, opts) {
    const box = clear(container);
    const list = el('div', { cls: 'hbars' });
    if (!rows.length) {
      box.appendChild(el('p', { cls: 'empty', text: '还没有数据' }));
      return;
    }
    rows.forEach((r) => {
      const fill = el('span', { cls: 'hbar-fill' + (r.cls ? ' ' + r.cls : '') });
      width(fill, r.value / opts.max);
      const track = el('span', { cls: 'hbar-track' }, [fill]);
      if (opts.mark !== undefined) {
        const mark = el('span', { cls: 'hbar-mark', attrs: { title: opts.markLabel || '' } });
        left(mark, opts.mark / opts.max);
        track.appendChild(mark);
      }
      const row = el('div', { cls: 'hbar', attrs: { tabindex: 0, role: 'img', 'aria-label': r.label + ' ' + r.text } }, [
        el('span', { cls: 'row-label', text: r.label }),
        track,
        el('span', { cls: 'hbar-value', text: r.text }),
      ]);
      attachTip(row, () => [r.text, r.label].concat(r.tip ? [r.tip] : []));
      list.appendChild(row);
    });
    box.appendChild(list);
  }

  function weeklyChart(container, weeks, target) {
    const box = clear(container);
    const W = Math.max(300, Math.round(box.clientWidth || 720));
    const H = W < 520 ? 220 : 260;
    const pad = { l: 44, r: 24, t: 20, b: 38 };
    const iw = W - pad.l - pad.r;
    const ih = H - pad.t - pad.b;
    const x = (i) => pad.l + (weeks.length === 1 ? iw / 2 : (i * iw) / (weeks.length - 1));
    const y = (v) => pad.t + (1 - v) * ih;
    const title = '每周采纳率（按推荐组）：' + weeks.map((w) => w.label + ' ' + (w.shadow ? '影子模式' : pct(w.adoption_rate, 0))).join('，');
    const chart = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': title });
    chart.appendChild(svgEl('title', {}, [title]));
    weeks.forEach((w, i) => {
      if (!w.shadow) return;
      const half = weeks.length > 1 ? iw / (weeks.length - 1) / 2 : iw / 2;
      chart.appendChild(svgEl('rect', { class: 'shadow-band', x: Math.max(pad.l, x(i) - half), y: pad.t, width: half * (i === 0 ? 1 : 2), height: ih, rx: 6 }));
      chart.appendChild(svgEl('text', { class: 'axis-text', x: Math.max(pad.l + 6, x(i) - half + 8), y: pad.t + 16 }, ['影子模式']));
    });
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      chart.appendChild(svgEl('line', { class: v === 0 ? 'axis-line' : 'grid-line', x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }));
      chart.appendChild(svgEl('text', { class: 'axis-text', x: pad.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, [Math.round(v * 100) + '%']));
    });
    chart.appendChild(svgEl('line', { class: 'target-line', x1: pad.l, x2: W - pad.r, y1: y(target), y2: y(target) }));
    // 目标线的标签放在线下方（采纳率曲线在目标线上方时不会压字）
    chart.appendChild(svgEl('text', { class: 'target-text', x: W - pad.r, y: y(target) + 16, 'text-anchor': 'end' }, ['目标 ' + pct(target, 0)]));
    weeks.forEach((w, i) => {
      const anchor = i === 0 ? 'start' : i === weeks.length - 1 ? 'end' : 'middle';
      const label = W < 520 ? w.label.split('–')[0] : w.label;
      chart.appendChild(svgEl('text', { class: 'axis-text', x: i === 0 ? pad.l - 4 : x(i), y: H - 12, 'text-anchor': anchor }, [label]));
    });
    const pts = weeks.map((w, i) => ({ w, i })).filter((p) => p.w.adoption_rate !== null && p.w.adoption_rate !== undefined);
    if (pts.length) {
      const line = pts.map((p, j) => (j ? 'L' : 'M') + x(p.i).toFixed(1) + ' ' + y(p.w.adoption_rate).toFixed(1)).join(' ');
      const area = line + ' L' + x(pts[pts.length - 1].i).toFixed(1) + ' ' + y(0) + ' L' + x(pts[0].i).toFixed(1) + ' ' + y(0) + ' Z';
      chart.appendChild(svgEl('path', { class: 'series-area', d: area }));
      chart.appendChild(svgEl('path', { class: 'series-line', d: line }));
      pts.forEach((p, j) => {
        const hit = svgEl('circle', { class: 'hit', cx: x(p.i), cy: y(p.w.adoption_rate), r: 14, tabindex: 0, 'aria-label': p.w.label + ' 采纳率 ' + pct(p.w.adoption_rate, 1) });
        attachTip(hit, () => [pct(p.w.adoption_rate, 1), '第 ' + p.w.week + ' 周 · ' + p.w.label, p.w.groups + ' 组推荐 · 原样 ' + pct(p.w.as_is_rate, 0)]);
        chart.appendChild(hit);
        chart.appendChild(svgEl('circle', { class: 'series-dot', cx: x(p.i), cy: y(p.w.adoption_rate), r: 5 }));
        if (j === pts.length - 1 || j === 0) {
          chart.appendChild(svgEl('text', { class: 'value-text', x: x(p.i), y: y(p.w.adoption_rate) - 12, 'text-anchor': 'middle' }, [pct(p.w.adoption_rate, 0)]));
        }
      });
    }
    box.appendChild(el('div', { cls: 'legend' }, [
      el('span', {}, [el('span', { cls: 'legend-key is-line' }), '采纳率']),
      el('span', {}, [el('span', { cls: 'legend-key is-band' }), '影子模式（不展示，不计采纳率）']),
    ]));
    box.appendChild(chart);
    const table = el('table', {}, [
      el('thead', {}, [el('tr', {}, [el('th', { text: '周' }), el('th', { text: '推荐组' }), el('th', { text: '采纳率' }), el('th', { text: '原样采纳率' })])]),
      el('tbody', {}, weeks.map((w) => el('tr', {}, [el('td', { text: w.label }), el('td', { text: w.shadow ? '影子 ' + w.shadow_records + ' 条' : String(w.groups) }), el('td', { text: pct(w.adoption_rate, 1) }), el('td', { text: pct(w.as_is_rate, 1) })]))),
    ]);
    box.appendChild(el('details', { cls: 'data-table' }, [el('summary', { text: '查看数据表' }), table]));
  }

  function qaPanel(container, qa) {
    const box = clear(container);
    box.appendChild(
      el('div', { cls: 'stat-row' }, [
        el('div', {}, [el('span', { cls: 'stat-big', text: num(qa.messages_intercepted) }), el('span', { cls: 'small muted', text: '条消息被拦截（共检查 ' + num(qa.messages_checked) + ' 条）' })]),
        el('div', {}, [el('span', { cls: 'stat-big', text: num(qa.drafts_failed) + ' / ' + num(qa.drafts_checked) }), el('span', { cls: 'small muted', text: '份草稿不通过 / 质检次数' })]),
      ])
    );
    const rows = qa.hard_hits.map((h) => ({ label: h.label, value: h.count, text: h.count + ' 次', cls: h.blocking ? 'is-bad' : 'is-muted', tip: h.blocking ? '命中就拦截' : '自动修正，不拦截' }));
    const holder = el('div');
    box.appendChild(holder);
    hbars(holder, rows, { max: Math.max(1, ...qa.hard_hits.map((h) => h.count)) });
    if (qa.top_reasons.length) {
      box.appendChild(el('p', { cls: 'small muted', text: '最常见的原因：' + qa.top_reasons.slice(0, 2).map((r) => r.reason + '（' + r.count + '）').join('；') }));
    }
  }

  function latencyPanel(container, lat) {
    const box = clear(container);
    const max = Math.max(1, ...lat.stages.map((s) => s.p95 || 0), lat.e2e.p95 || 0);
    const rows = lat.stages.map((s) => ({ label: s.label, value: s.p95 || 0, text: ms(s.p95), tip: 'P50 ' + ms(s.p50) + ' · P95 ' + ms(s.p95) }));
    rows.push({ label: '端到端', value: lat.e2e.p95 || 0, text: ms(lat.e2e.p95), tip: 'P50 ' + ms(lat.e2e.p50) + ' · P95 ' + ms(lat.e2e.p95) + ' · 目标 ≤ 5 秒' });
    box.appendChild(el('div', { cls: 'legend' }, [el('span', {}, [el('span', { cls: 'legend-key' }), 'P95（悬停看 P50）'])]));
    const holder = el('div');
    box.appendChild(holder);
    hbars(holder, rows, { max });
    const within = (lat.e2e.p95 || 0) <= lat.target_ms;
    box.appendChild(el('p', { cls: 'small ' + (within ? 'muted' : 'risk'), text: (within ? '✓ 端到端 P95 ' : '✗ 端到端 P95 ') + ms(lat.e2e.p95) + '，目标 ≤ 5 秒。' + (lat.offline ? '离线替身不联网；接真实大模型后按 §12.1 各环节目标核对。' : '') }));
  }

  function metricsPanel(container, pm) {
    const box = clear(container);
    $('metrics-sub').textContent = '接入日 ' + pm.go_live + '，前后各 4 周的平均（§12.4）· 数据来自平台指标接口（合成）';
    if (!pm.metrics.length) {
      box.appendChild(el('p', { cls: 'empty', text: '没有平台指标数据' }));
      return;
    }
    pm.metrics.forEach((m) => {
      const max = Math.max(m.before, m.after) * 1.15 || 1;
      const rel = m.before ? (m.after - m.before) / m.before : 0;
      const arrow = m.delta >= 0 ? '▲' : '▼';
      const delta = el('span', { cls: 'delta ' + (m.improved ? 'delta-good' : 'delta-bad'), text: arrow + ' ' + (m.unit === 'ratio' ? (m.delta * 100).toFixed(1) + ' 个百分点' : Math.abs(rel * 100).toFixed(0) + '%') + ' · ' + (m.improved ? '改善' : '变差') });
      const holder = el('div');
      const card = el('article', { cls: 'metric', attrs: { 'data-metric': m.metric } }, [
        el('div', { cls: 'metric-head' }, [el('span', { cls: 'metric-label', text: m.label }), delta]),
        el('div', { cls: 'legend' }, [el('span', {}, [el('span', { cls: 'legend-key is-muted' }), '接入前']), el('span', {}, [el('span', { cls: 'legend-key' }), '接入后']), el('span', { text: m.higher_is_better ? '越高越好' : '越低越好' })]),
        holder,
        el('p', { cls: 'metric-note', text: m.note }),
      ]);
      hbars(
        holder,
        [
          { label: '接入前', value: m.before, text: metricValue(m.before, m.unit), cls: 'is-muted', tip: m.days_before + ' 天平均' },
          { label: '接入后', value: m.after, text: metricValue(m.after, m.unit), tip: m.days_after + ' 天平均' },
        ],
        { max }
      );
      box.appendChild(card);
    });
  }

  // ================================================================ 合规：打码演示

  const MASK_TOKEN = /(\[(?:手机号|地址|证件号|邮箱|卡号|联系方式)\])/;

  $('mask-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = $('mask-input').value.trim();
    if (!text) {
      toast('先输入一段文字', true);
      return;
    }
    try {
      const out = await api('/mask', { method: 'POST', body: { text } });
      const output = clear($('mask-output'));
      out.masked.split(MASK_TOKEN).forEach((piece) => {
        if (!piece) return;
        output.appendChild(MASK_TOKEN.test(piece) ? el('mark', { text: piece }) : document.createTextNode(piece));
      });
      const found = clear($('mask-found'));
      if (!out.found.length) found.appendChild(el('span', { cls: 'chip', text: '没有需要遮的个人信息' }));
      out.found.forEach((f) => found.appendChild(el('span', { cls: 'chip', text: '遮掉 ' + f.label + ' × ' + f.count })));
    } catch (err) {
      fail(err);
    }
  });

  // ================================================================ 重置

  $('reset').addEventListener('click', async () => {
    const platform = state.platform;
    const answer = await confirmDialog({
      title: '重置' + PLATFORMS[platform].name + '演示？',
      lead: '重新组装这个平台的演示单元、重新回放近 5 周的合成数据；工作台和售后审核里做过的操作都会清空。',
      details: [
        ['店铺', PLATFORMS[platform].shop_name + '（合成）'],
        ['影响', '只影响当前平台'],
      ],
      ok: '重置',
    });
    if (!answer) return;
    try {
      state.loaded[platform] = false;
      await withLoading(platform, () => api('/reset', { method: 'POST', body: { platform } }));
      delete state.scenarios[platform];
      delete state.aftersales[platform];
      delete state.active[platform];
      delete state.selected[platform];
      Object.keys(state.results).forEach((k) => {
        if (k.indexOf(platform + ':') === 0) delete state.results[k];
      });
      toast('已重置：' + PLATFORMS[platform].name + '演示回到初始状态');
      loadRoute();
    } catch (err) {
      fail(err);
    }
  });

  // ================================================================ 启动

  $('footer-text').textContent = CONFIG.footer;
  renderPlatformSwitch();
  applyPlatform();
  window.addEventListener('hashchange', onHashChange);
  showRoute(routeFromHash());
  loadRoute();
})();
