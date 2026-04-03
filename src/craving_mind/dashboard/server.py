"""FastAPI/WebSocket dashboard server for CravingMind."""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from craving_mind.dashboard.metrics import MetricsCollector
from craving_mind.dashboard.storage import MetricsStorage

# ---------------------------------------------------------------------------
# HTML Template (single-file, embedded CSS + JS)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CravingMind Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  /* ---- Reset & Base ---- */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0d0d1a;
    --bg2:      #12122a;
    --bg3:      #1a1a3e;
    --panel:    #16163a;
    --border:   #2a2a5a;
    --accent:   #e94560;
    --success:  #16c79a;
    --warning:  #f7b731;
    --danger:   #e94560;
    --info:     #4a90d9;
    --muted:    #6b6b9b;
    --text:     #e0e0f0;
    --text2:    #a0a0c0;
    --mono:     'Consolas', 'Monaco', 'Courier New', monospace;
  }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; min-height: 100vh; display: flex; flex-direction: column; }

  /* ---- Top Bar ---- */
  #topbar {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 8px 18px;
    display: flex; align-items: center; gap: 18px;
    position: sticky; top: 0; z-index: 100;
    min-height: 44px;
  }
  .logo { font-size: 18px; font-weight: 800; color: var(--accent); letter-spacing: 1px; text-transform: uppercase; }
  .logo span { color: var(--text2); font-weight: 400; }
  .tb-sep { width: 1px; background: var(--border); height: 24px; }
  .tb-item { display: flex; flex-direction: column; align-items: center; }
  .tb-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  .tb-val { font-size: 14px; font-weight: 700; }
  .tb-val.accent { color: var(--accent); }
  .phase-badge { padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
  .phase-1 { background: #1a3a5a; color: #4a9fd9; }
  .phase-2 { background: #3a2a0a; color: #f7b731; }
  .phase-3 { background: #3a0a0a; color: #e94560; }
  .ws-dot { width: 10px; height: 10px; border-radius: 50%; background: #e94560; transition: background .4s; }
  .ws-dot.connected { background: var(--success); box-shadow: 0 0 6px var(--success); }
  .ctrl-btn { background: var(--bg3); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 3px 10px; font-size: 12px; cursor: pointer; margin-right: 6px; transition: all .2s; }
  .ctrl-btn:hover { background: var(--accent); color: #fff; }
  .ctrl-btn.ctrl-stop:hover { background: #e94560; }
  .ctrl-btn.active { background: var(--warn); color: #000; border-color: var(--warn); }
  .ctrl-btn.active.ctrl-stop { background: #e94560; color: #fff; border-color: #e94560; }
  .spacer { flex: 1; }
  .ts-display { font-size: 11px; color: var(--muted); font-family: var(--mono); }

  /* ---- Main Grid ---- */
  #main {
    flex: 1;
    display: grid;
    grid-template-columns: 240px 1fr 240px;
    grid-template-rows: 1fr;
    gap: 10px;
    padding: 10px;
    padding-right: 10px;
    transition: padding-right 0.3s ease;
  }

  /* ---- Panels ---- */
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    overflow: hidden;
  }
  .panel-title {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--muted); margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .panel-title::after { content: ''; flex: 1; height: 1px; background: var(--border); }

  /* ---- Stat Rows ---- */
  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid rgba(42,42,90,0.5); }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: var(--text2); }
  .stat-val { font-weight: 700; font-family: var(--mono); }
  .stat-val.good { color: var(--success); }
  .stat-val.warn { color: var(--warning); }
  .stat-val.bad { color: var(--danger); }
  .stat-val.info { color: var(--info); }

  /* ---- Progress Bars ---- */
  .bar-wrap { margin: 8px 0; }
  .bar-label { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px; color: var(--text2); }
  .bar-track { height: 8px; background: var(--bg3); border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width .6s ease; }
  .bar-budget .bar-fill { background: linear-gradient(90deg, var(--success), #0aa07a); }
  .bar-budget.warn .bar-fill { background: linear-gradient(90deg, var(--warning), #e0a020); }
  .bar-budget.crit .bar-fill { background: linear-gradient(90deg, var(--danger), #c03040); }
  .bar-bible .bar-fill { background: linear-gradient(90deg, #4a5ae0, #2a3aaa); }

  /* ---- Charts ---- */
  #charts-panel { grid-column: 2; grid-row: 1 / 3; }
  .chart-wrap { position: relative; height: 160px; margin-bottom: 14px; }
  .chart-wrap:last-child { margin-bottom: 0; }
  .chart-title { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }

  /* ---- Left / Right Panels ---- */
  #left-panel { grid-column: 1; grid-row: 1 / 3; display: flex; flex-direction: column; gap: 10px; }
  #right-panel { grid-column: 3; grid-row: 1 / 3; display: flex; flex-direction: column; gap: 10px; }

  /* ---- Score bars (right panel) ---- */
  .score-type-row { margin: 4px 0; }
  .score-type-label { display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px; }
  .score-bar-track { height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .score-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--info), #2a70b9); transition: width .6s; }

  /* ---- Artifacts version list ---- */
  .version-list { max-height: 120px; overflow-y: auto; }
  .version-row { display: flex; justify-content: space-between; font-size: 11px; padding: 2px 4px; border-radius: 3px; cursor: pointer; }
  .version-row:hover { background: var(--bg3); }
  .version-row.best { color: var(--success); font-weight: 700; }

  /* ---- Log Panel ---- */
  #log-panel {
    flex: 1;
    min-height: 180px;
    display: flex;
    flex-direction: column;
  }
  .log-inner { flex: 1; min-height: 140px; max-height: 400px; overflow-y: auto; font-family: var(--mono); font-size: 11px; }
  .log-entry { padding: 2px 6px; border-radius: 3px; margin: 1px 0; line-height: 1.5; }
  .log-entry .ts { color: var(--muted); margin-right: 8px; }
  .log-entry .ep { color: var(--info); margin-right: 8px; }
  .log-pass { color: var(--success); }
  .log-fail { color: var(--danger); }
  .log-warn { color: var(--warning); }
  .log-info { color: var(--text2); }
  .log-oom  { color: var(--danger); font-weight: 700; }
  .log-drift { color: var(--warning); font-weight: 700; }
  .log-artifact { color: #c084fc; }

  /* ---- File Viewer (inside console panel) ---- */
  .fv-header { display: flex; align-items: center; gap: 0; flex-wrap: wrap; }
  .fv-tab {
    padding: 5px 14px; font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; cursor: pointer; border: 1px solid var(--border);
    border-bottom: none; background: var(--bg3); color: var(--muted);
    border-radius: 6px 6px 0 0; margin-right: 4px; transition: background .2s;
  }
  .fv-tab:hover { background: var(--bg2); color: var(--text); }
  .fv-tab.active { background: var(--panel); color: var(--accent); border-color: var(--accent); }
  #fv-version-select { background: var(--bg3); color: var(--text); border: 1px solid var(--border); padding: 2px 6px; border-radius: 4px; font-size: 11px; }
  .fv-scroll { overflow: auto; padding: 10px; }
  .fv-content { font-family: var(--mono); font-size: 12px; line-height: 1.6; white-space: pre; tab-size: 4; }
  /* Markdown */
  .md-h1 { color: var(--accent); font-size: 16px; font-weight: 800; margin: 8px 0 4px; font-family: var(--mono); }
  .md-h2 { color: var(--info); font-size: 14px; font-weight: 700; margin: 6px 0 3px; font-family: var(--mono); }
  .md-h3 { color: var(--text); font-size: 13px; font-weight: 600; font-family: var(--mono); }
  .md-bold { color: var(--warning); font-weight: 700; }
  .md-code { background: var(--bg3); color: var(--success); padding: 0 4px; border-radius: 3px; }
  .md-codeblock { display: block; background: var(--bg3); color: var(--success); padding: 8px; border-radius: 4px; margin: 6px 0; white-space: pre; }
  .md-li::before { content: '• '; color: var(--accent); }
  /* Python syntax */
  .py-kw { color: #cc99cd; }
  .py-str { color: #7ec699; }
  .py-cmt { color: #6b7280; font-style: italic; }
  .py-num { color: #f9a825; }
  .py-fn { color: #80c8f4; }
  .py-deco { color: var(--warning); }
  .py-ln { color: var(--muted); user-select: none; display: inline-block; width: 36px; text-align: right; margin-right: 10px; padding-right: 6px; border-right: 1px solid var(--border); }

  /* Collapse toggle */
  .collapse-btn { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 16px; padding: 0 4px; }
  .collapse-btn:hover { color: var(--text); }

  /* ---- Scrollbars ---- */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

  /* ---- Pulse animation for OOM ---- */
  @keyframes pulse-red { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .oom-alert { animation: pulse-red 1s infinite; }

  /* ---- Responsive tweak ---- */
  @media (max-width: 1100px) {
    #main { grid-template-columns: 200px 1fr 200px; }
  }

  /* ---- Live Console Panel (fixed, right edge) ---- */
  #console-panel {
    position: fixed;
    top: 44px;
    right: 0;
    bottom: 0;
    width: 480px;
    z-index: 60;
    background: #0b0b18;
    border-left: 2px solid var(--border);
    display: flex;
    flex-direction: column;
    transition: transform 0.3s ease;
    box-shadow: -4px 0 20px rgba(0,0,0,0.5);
  }
  #console-panel.collapsed {
    transform: translateX(100%);
  }
  #console-toggle-btn {
    position: absolute;
    left: -28px;
    top: 50%;
    transform: translateY(-50%);
    width: 28px;
    height: 56px;
    background: #1a1a3e;
    border: 1px solid var(--border);
    border-right: none;
    border-radius: 6px 0 0 6px;
    color: var(--text2);
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    writing-mode: vertical-rl;
    letter-spacing: 1px;
    font-weight: 700;
    transition: background 0.2s, color 0.2s;
  }
  #console-toggle-btn:hover { background: var(--bg3); color: var(--text); }
  #console-header {
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    background: var(--bg2);
  }
  #console-header .fv-header {
    padding: 6px 10px 0;
    margin-bottom: 0;
  }
  #console-header .fv-tab {
    font-size: 10px;
    padding: 4px 10px;
  }
  .cp-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 10px;
  }
  #console-file-body {
    flex: 1;
    overflow: auto;
    padding: 6px 8px;
  }
  #console-file-body .fv-scroll {
    max-height: none;
    height: 100%;
  }
  .console-autoscroll-label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
  }
  .console-autoscroll-label input[type=checkbox] {
    accent-color: var(--success);
    cursor: pointer;
  }
  #console-clear-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 1px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 10px;
  }
  #console-clear-btn:hover { color: var(--text); border-color: var(--muted); }
  #console-body {
    flex: 1;
    overflow-y: auto;
    overflow-x: auto;
    padding: 6px 8px;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.6;
    white-space: pre;
  }
  .cl-pass  { color: #16c79a; }
  .cl-fail  { color: #e94560; }
  .cl-warn  { color: #f7b731; }
  .cl-info  { color: #6b8fa8; }
  .cl-epoch { color: #c084fc; font-weight: 700; }
  .cl-tool  { color: #3d6a78; padding-left: 4px; }
  .cl-crav  { color: #5a5a82; font-style: italic; padding-left: 4px; cursor: pointer; }
  .cl-judge { color: #5a6a7a; padding-left: 4px; }
  .crav-toggle { color: #4a90d9; font-style: normal; user-select: none; font-size: 10px; margin-left: 4px; }
  .cl-line  { padding: 1px 2px; border-radius: 2px; }
  .cl-line:hover { background: rgba(255,255,255,0.04); }

  /* ---- Console drag handle ---- */
  #console-drag-handle {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    cursor: col-resize;
    z-index: 2;
  }
  #console-drag-handle:hover, #console-drag-handle.dragging {
    background: var(--border);
  }
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <div class="logo">Craving<span>Mind</span></div>
  <div class="tb-sep"></div>
  <div class="tb-item">
    <span class="tb-label">Crav ID</span>
    <span class="tb-val accent" id="tb-crav-id">—</span>
  </div>
  <div class="tb-sep"></div>
  <div class="tb-item">
    <span class="tb-label">Phase</span>
    <span id="tb-phase" class="phase-badge phase-1">Phase 1</span>
  </div>
  <div class="tb-sep"></div>
  <div class="tb-item">
    <span class="tb-label">Epoch</span>
    <span class="tb-val" id="tb-epoch">—</span>
  </div>
  <div class="tb-sep"></div>
  <div class="tb-item">
    <span class="tb-label">Success Rate</span>
    <span class="tb-val" id="tb-sr">—</span>
  </div>
  <div class="tb-sep"></div>
  <div class="tb-item">
    <span class="tb-label">Best Score</span>
    <span class="tb-val" id="tb-best">—</span>
  </div>
  <div class="spacer"></div>
  <button id="btn-pause" class="ctrl-btn" onclick="ctrlPause()" title="Pause after current task">⏸ Pause</button>
  <button id="btn-stop" class="ctrl-btn ctrl-stop" onclick="ctrlStop()" title="Stop after current epoch">⏹ Stop</button>
  <span class="ts-display" id="tb-ts">—</span>
  <div class="ws-dot" id="ws-dot" title="WebSocket"></div>
</div>

<!-- MAIN GRID -->
<div id="main">

  <!-- LEFT PANEL: Health & Status -->
  <div id="left-panel">

    <div class="panel">
      <div class="panel-title">Health</div>

      <div class="bar-wrap">
        <div class="bar-label"><span>Budget Remaining</span><span id="budget-pct">—</span></div>
        <div class="bar-track"><div class="bar-fill" id="budget-bar" style="width:0%"></div></div>
      </div>

      <div class="bar-wrap">
        <div class="bar-label"><span>Bible Weight</span><span id="bible-pct">—</span></div>
        <div class="bar-track bar-bible"><div class="bar-fill bar-bible" id="bible-bar" style="width:0%"></div></div>
      </div>

      <div class="stat-row">
        <span class="stat-label">OOM Events</span>
        <span class="stat-val" id="oom-count">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Starvation Rate</span>
        <span class="stat-val" id="starv-rate">0%</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Saved Tokens</span>
        <span class="stat-val info" id="saved-tok">—</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Phase Progress</div>
      <div class="stat-row">
        <span class="stat-label">Phase 1</span>
        <span class="stat-val info" style="font-size:11px">Epochs 1–10</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Phase 2</span>
        <span class="stat-val info" style="font-size:11px">Epochs 11–25</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Phase 3</span>
        <span class="stat-val info" style="font-size:11px">Epoch 26+</span>
      </div>
      <div class="bar-wrap" style="margin-top:8px">
        <div class="bar-label"><span>Phase Progress</span><span id="phase-pct-txt">—</span></div>
        <div class="bar-track"><div class="bar-fill" id="phase-bar" style="width:0%; background: linear-gradient(90deg,#4a5ae0,#e94560)"></div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Efficiency</div>
      <div class="stat-row">
        <span class="stat-label">Cost / Pass</span>
        <span class="stat-val" id="cost-pass">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Success Rate (10ep)</span>
        <span class="stat-val good" id="eff-sr">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Frozen SR</span>
        <span class="stat-val info" id="frozen-sr">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Dynamic SR</span>
        <span class="stat-val" id="dyn-sr">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Overfit Gap</span>
        <span class="stat-val warn" id="overfit-gap">—</span>
      </div>
    </div>

  </div>

  <!-- CENTER: Charts -->
  <div id="charts-panel" class="panel">
    <div class="panel-title">Performance Over Epochs</div>

    <div class="chart-title">Success Rate — Frozen vs Dynamic</div>
    <div class="chart-wrap"><canvas id="chart-sr"></canvas></div>

    <div class="chart-title">Semantic vs Entity Score</div>
    <div class="chart-wrap"><canvas id="chart-scores"></canvas></div>

    <div class="chart-title">Overfit Gap (Frozen − Dynamic)</div>
    <div class="chart-wrap" style="height:100px"><canvas id="chart-overfit"></canvas></div>

    <div class="chart-title">Saved Tokens per Epoch</div>
    <div class="chart-wrap" style="height:100px"><canvas id="chart-tokens"></canvas></div>
  </div>

  <!-- RIGHT PANEL: Artifacts & Evolution -->
  <div id="right-panel">

    <div class="panel">
      <div class="panel-title">Artifact</div>
      <div class="stat-row">
        <span class="stat-label">Latest Version</span>
        <span class="stat-val accent" id="art-version">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Best Score</span>
        <span class="stat-val good" id="art-best-score">—</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Best Epoch</span>
        <span class="stat-val" id="art-best-epoch">—</span>
      </div>
      <div style="margin-top:8px">
        <div class="panel-title" style="margin-bottom:6px">Score by Type</div>
        <div id="score-by-type"></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Version History</div>
      <div class="version-list" id="version-list"></div>
    </div>

    <div class="panel">
      <div class="panel-title">Evolution</div>
      <div class="stat-row">
        <span class="stat-label">Compressions</span>
        <span class="stat-val" id="evo-compressions">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Drift Events</span>
        <span class="stat-val warn" id="evo-drift">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Graveyard Size</span>
        <span class="stat-val" id="evo-graveyard">0</span>
      </div>
    </div>

    <!-- LOG PANEL (right column) -->
    <div id="log-panel" class="panel">
      <div class="panel-title">
        Live Event Log
        <button onclick="clearLog()" style="background:none;border:1px solid var(--border);color:var(--muted);padding:1px 8px;border-radius:4px;cursor:pointer;font-size:10px;margin-left:8px">Clear</button>
      </div>
      <div class="log-inner" id="log-inner"></div>
    </div>

  </div>


</div><!-- /main -->

<!-- LIVE CONSOLE PANEL (fixed, far right, collapsible) -->
<div id="console-panel">
  <div id="console-drag-handle" title="Drag to resize"></div>
  <button id="console-toggle-btn" onclick="consoleToggle()" title="Collapse Live Console">▶</button>
  <div id="console-header">
    <div class="fv-header">
      <span class="fv-tab active" data-tab="console" onclick="cpSelectTab(this)">Console</span>
      <span class="fv-tab" data-tab="compress.py" onclick="cpSelectTab(this)">compress.py <span id="fv-ver-compress" style="font-size:10px;opacity:0.6"></span></span>
      <span class="fv-tab" data-tab="bible.md" onclick="cpSelectTab(this)">bible.md</span>
      <span class="fv-tab" data-tab="graveyard.md" onclick="cpSelectTab(this)">graveyard.md</span>
      <span class="fv-tab" data-tab="artifact" onclick="cpSelectTab(this)">Artifact</span>
    </div>
    <div id="console-controls" class="cp-controls">
      <label class="console-autoscroll-label" title="Auto-scroll to newest lines">
        <input type="checkbox" id="console-autoscroll" checked>
        Auto-scroll
      </label>
      <label class="console-autoscroll-label" title="Wrap long lines">
        <input type="checkbox" id="console-wrap">
        Wrap
      </label>
      <button id="console-clear-btn" onclick="consoleClear()">Clear</button>
    </div>
    <div id="file-controls" class="cp-controls" style="display:none">
      <select id="fv-version-select" onchange="fvLoadArtifact()" style="display:none">
        <option value="">— version —</option>
      </select>
      <label class="console-autoscroll-label">
        <input type="checkbox" id="fv-auto-refresh"> Auto-refresh
      </label>
      <button id="fv-refresh-btn" onclick="fvRefresh()" style="background:var(--accent);color:#fff;border:none;padding:2px 10px;border-radius:4px;font-size:10px;cursor:pointer;font-weight:600">↻</button>
    </div>
  </div>
  <div id="console-body"></div>
  <div id="console-file-body" style="display:none">
    <div class="fv-scroll">
      <div class="fv-content" id="fv-content"><span style="color:var(--muted)">Select a file tab to view content.</span></div>
    </div>
  </div>
</div>

<script>
// ============================================================
// WebSocket
// ============================================================
let ws = null;
let wsRetryTimer = null;
let state = {};
let logEntries = [];
let seenEpochs = new Set();
let fvCurrentFile = 'compress.py';
let fvAutoTimer = null;
let artifactVersions = [];
let cpActiveTab = 'console';
// Track content hashes per file to detect actual changes.
const fvHashes = {};     // filename → last seen hash
const fvVersions = {};   // filename → version counter (1-based)

// ============================================================
// Run Control (pause / stop)
// ============================================================
function ctrlPause() {
  const btn = document.getElementById('btn-pause');
  const isPaused = btn.classList.toggle('active');
  btn.textContent = isPaused ? '▶ Resume' : '⏸ Pause';
  fetch('/api/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: isPaused ? 'pause' : 'resume'}),
  });
}

function ctrlStop() {
  if (!confirm('Stop the run after current epoch finishes?')) return;
  const btn = document.getElementById('btn-stop');
  btn.classList.add('active');
  btn.textContent = '⏹ Stopping…';
  btn.disabled = true;
  fetch('/api/control', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'stop'}),
  });
}

// Update button states from server control state.
function updateControlButtons(ctrl) {
  if (!ctrl) return;
  const pauseBtn = document.getElementById('btn-pause');
  const stopBtn = document.getElementById('btn-stop');
  if (ctrl.paused) {
    pauseBtn.classList.add('active');
    pauseBtn.textContent = '▶ Resume';
  } else {
    pauseBtn.classList.remove('active');
    pauseBtn.textContent = '⏸ Pause';
  }
  if (ctrl.stopped) {
    stopBtn.classList.add('active');
    stopBtn.textContent = '⏹ Stopping…';
    stopBtn.disabled = true;
  }
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('ws-dot').classList.add('connected');
    if (wsRetryTimer) { clearTimeout(wsRetryTimer); wsRetryTimer = null; }
  };

  ws.onclose = () => {
    document.getElementById('ws-dot').classList.remove('connected');
    wsRetryTimer = setTimeout(connect, 3000);
  };

  ws.onmessage = (ev) => {
    try { state = JSON.parse(ev.data); } catch(e) { return; }
    render(state);
  };
}

// ============================================================
// Render
// ============================================================
function render(s) {
  if (!s) return;
  const live = s.live || {};
  const health = s.health || {};
  const eff = s.efficiency || {};
  const evo = s.evolution || {};
  const art = s.artifact || {};
  const hist = s.epoch_history || [];
  const ctrl = s.control || {};

  // Control buttons
  updateControlButtons(ctrl);

  // Top bar
  document.getElementById('tb-epoch').textContent = live.current_epoch ?? '—';
  document.getElementById('tb-sr').textContent = live.latest_success_rate != null ? live.latest_success_rate + '%' : '—';
  document.getElementById('tb-best').textContent = art.best_mean_score != null ? (art.best_mean_score*100).toFixed(1)+'%' : '—';
  document.getElementById('tb-ts').textContent = new Date().toLocaleTimeString();
  document.getElementById('tb-crav-id').textContent = live.crav_id || 'Crav-???';

  const phase = live.phase || health.phase || 1;
  const phaseBadge = document.getElementById('tb-phase');
  phaseBadge.textContent = 'Phase ' + phase;
  phaseBadge.className = 'phase-badge phase-' + phase;

  // Health
  const budgetPct = health.budget_remaining_pct ?? 0;
  const budgetBar = document.getElementById('budget-bar').parentElement;
  budgetBar.className = 'bar-track bar-budget' + (budgetPct < 15 ? ' crit' : budgetPct < 40 ? ' warn' : '');
  document.getElementById('budget-bar').style.width = budgetPct + '%';
  document.getElementById('budget-pct').textContent = budgetPct.toFixed(1) + '%';

  const biblePct = health.bible_weight_pct ?? 0;
  document.getElementById('bible-bar').style.width = Math.min(biblePct, 100) + '%';
  document.getElementById('bible-pct').textContent = biblePct.toFixed(1) + '%';

  const oomEl = document.getElementById('oom-count');
  oomEl.textContent = health.oom_count ?? 0;
  oomEl.className = 'stat-val' + ((health.oom_count > 0) ? ' bad oom-alert' : '');

  document.getElementById('starv-rate').textContent = (health.starvation_rate_pct ?? 0).toFixed(1) + '%';
  document.getElementById('saved-tok').textContent = fmt(health.saved_tokens ?? 0);

  // Phase bar
  const epochNum = live.current_epoch ?? 0;
  const maxEpoch = 50; // display range
  const phasePct = Math.min(100, (epochNum / maxEpoch) * 100);
  document.getElementById('phase-bar').style.width = phasePct + '%';
  document.getElementById('phase-pct-txt').textContent = 'Epoch ' + epochNum;

  // Efficiency
  document.getElementById('cost-pass').textContent = eff.cost_per_pass != null ? fmt(eff.cost_per_pass) : '—';
  setStatVal('eff-sr', (eff.success_rate_pct ?? 0) + '%', pctColor(eff.success_rate_pct ?? 0));
  document.getElementById('frozen-sr').textContent = (eff.frozen_success_rate_pct ?? 0) + '%';
  document.getElementById('dyn-sr').textContent = (eff.dynamic_success_rate_pct ?? 0) + '%';
  setStatVal('overfit-gap', (eff.mean_overfit_gap ?? 0).toFixed(1) + '%', eff.mean_overfit_gap > 10 ? 'warn' : '');

  // Artifact
  document.getElementById('art-version').textContent = art.latest_version_str ?? '—';
  document.getElementById('art-best-score').textContent = art.best_mean_score != null ? (art.best_mean_score*100).toFixed(1)+'%' : '—';
  document.getElementById('art-best-epoch').textContent = art.best_epoch ?? '—';

  // Score by type
  const sbt = art.score_by_type || {};
  const sbtEl = document.getElementById('score-by-type');
  sbtEl.innerHTML = '';
  for (const [k, v] of Object.entries(sbt)) {
    const pct = (v * 100).toFixed(1);
    sbtEl.innerHTML += `
      <div class="score-type-row">
        <div class="score-type-label"><span>${k}</span><span>${pct}%</span></div>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${pct}%"></div></div>
      </div>`;
  }

  // Version list
  const vl = document.getElementById('version-list');
  const versions = (art.versions || []).slice().reverse().slice(0, 20);
  vl.innerHTML = '';
  artifactVersions = art.versions || [];
  for (const v of versions) {
    const isBest = v.mean_score === art.best_mean_score;
    vl.innerHTML += `<div class="version-row ${isBest ? 'best' : ''}" onclick="fvViewArtifact(${v.version})">
      v${String(v.version).padStart(4,'0')} · ep${v.epoch} · ${(v.mean_score*100).toFixed(1)}%
    </div>`;
  }

  // Evolution
  document.getElementById('evo-compressions').textContent = evo.compressions_total ?? 0;
  document.getElementById('evo-drift').textContent = evo.drift_events ?? 0;
  document.getElementById('evo-graveyard').textContent = evo.graveyard_size ?? 0;

  // Charts
  updateCharts(hist, art);

  // Update artifact version dropdown
  updateVersionSelect(art.versions || []);

  // Log new epochs
  logNewEpochs(hist);

  // Live console
  consoleUpdate(s.console_lines || []);
}

// ============================================================
// Charts
// ============================================================
const CHART_DEFAULTS = {
  responsive: true, maintainAspectRatio: false,
  animation: { duration: 300 },
  plugins: { legend: { labels: { color: '#a0a0c0', font: { size: 10 } } } },
  scales: {
    x: { ticks: { color: '#6b6b9b', maxTicksLimit: 10, font: { size: 9 } }, grid: { color: '#1a1a3e' } },
    y: { ticks: { color: '#6b6b9b', font: { size: 9 } }, grid: { color: '#1a1a3e' } }
  }
};

function makeChart(id, type, data, opts) {
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, { type, data, options: { ...CHART_DEFAULTS, ...opts } });
}

let chartSR = null, chartScores = null, chartOverfit = null, chartTokens = null;

function ensureCharts() {
  if (!chartSR) {
    chartSR = makeChart('chart-sr', 'line', { labels: [], datasets: [
      { label: 'Frozen SR', data: [], borderColor: '#4a90d9', backgroundColor: 'rgba(74,144,217,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Dynamic SR', data: [], borderColor: '#16c79a', backgroundColor: 'rgba(22,199,154,0.1)', tension: 0.3, pointRadius: 2 },
    ]}, { scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1 } } });
  }
  if (!chartScores) {
    chartScores = makeChart('chart-scores', 'line', { labels: [], datasets: [
      { label: 'Semantic', data: [], borderColor: '#c084fc', backgroundColor: 'rgba(192,132,252,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Entity', data: [], borderColor: '#f7b731', backgroundColor: 'rgba(247,183,49,0.1)', tension: 0.3, pointRadius: 2 },
      { label: 'Artifact Best', data: [], borderColor: '#e94560', borderDash: [4,2], pointRadius: 3, pointStyle: 'star' },
    ]}, { scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1 } } });
  }
  if (!chartOverfit) {
    chartOverfit = makeChart('chart-overfit', 'bar', { labels: [], datasets: [
      { label: 'Overfit Gap', data: [], backgroundColor: (ctx) => ctx.raw > 0.1 ? 'rgba(233,69,96,0.7)' : 'rgba(74,144,217,0.5)' },
    ]}, {});
  }
  if (!chartTokens) {
    chartTokens = makeChart('chart-tokens', 'line', { labels: [], datasets: [
      { label: 'Saved Tokens', data: [], borderColor: '#16c79a', backgroundColor: 'rgba(22,199,154,0.15)', fill: true, tension: 0.4, pointRadius: 0 },
    ]}, {});
  }
}

function updateCharts(hist, art) {
  ensureCharts();
  const labels = hist.map(e => 'E' + e.epoch);
  const frozenSR = hist.map(e => e.frozen_success_rate);
  const dynSR   = hist.map(e => e.dynamic_success_rate);
  const overfit  = hist.map(e => e.overfit_gap);
  const tokens   = hist.map(e => e.saved_tokens);

  const sem = hist.map(e => e.semantic_score ?? 0);
  const ent = hist.map(e => e.entity_score ?? 0);

  // Artifact best score per epoch (sparse)
  const artBest = hist.map(e => e.artifact_path ? (art.best_mean_score ?? null) : null);

  setChartData(chartSR, labels, [frozenSR, dynSR]);
  setChartData(chartScores, labels, [sem, ent, artBest]);
  setChartData(chartOverfit, labels, [overfit]);
  setChartData(chartTokens, labels, [tokens]);
}

function setChartData(chart, labels, datasets) {
  chart.data.labels = labels;
  datasets.forEach((d, i) => { if (chart.data.datasets[i]) chart.data.datasets[i].data = d; });
  chart.update('none');
}

// ============================================================
// Event Log
// ============================================================
function logNewEpochs(hist) {
  for (const e of hist) {
    if (seenEpochs.has(e.epoch)) continue;
    seenEpochs.add(e.epoch);

    const ts = e.ts ? e.ts.substring(11, 19) : '—';
    let cls = 'log-info', msg = '';

    if (e.is_oom) {
      cls = 'log-oom';
      msg = `OOM — epoch ${e.epoch} aborted, budget exhausted`;
    } else if (e.drift_detected) {
      cls = 'log-drift';
      msg = `CUSUM drift detected (SR=${pct(e.success_rate)}) — R&D suppressed next epoch`;
    } else if (e.artifact_path) {
      cls = 'log-artifact';
      msg = `Artifact exported — SR=${pct(e.success_rate)}, frozen=${pct(e.frozen_success_rate)}`;
    } else if (e.success_rate >= 0.85) {
      cls = 'log-pass';
      msg = `PASS — SR=${pct(e.success_rate)}, frozen=${pct(e.frozen_success_rate)}, dynamic=${pct(e.dynamic_success_rate)}`;
    } else {
      cls = 'log-fail';
      msg = `FAIL — SR=${pct(e.success_rate)}, gap=${pct(e.overfit_gap)}`;
    }

    addLog(ts, e.epoch, cls, msg);
  }
}

function addLog(ts, epoch, cls, msg) {
  logEntries.push({ ts, epoch, cls, msg });
  if (logEntries.length > 200) logEntries.shift();
  renderLog();
}

function renderLog() {
  const el = document.getElementById('log-inner');
  el.innerHTML = logEntries.slice(-80).map(e =>
    `<div class="log-entry ${e.cls}"><span class="ts">${e.ts}</span><span class="ep">ep${e.epoch}</span>${escHtml(e.msg)}</div>`
  ).join('');
  el.scrollTop = el.scrollHeight;
}

function clearLog() { logEntries = []; seenEpochs = new Set(); renderLog(); }

// ============================================================
// Live Console
// ============================================================
let consoleCollapsed = false;
let consoleLines = [];
let consoleWidth = 480;

function consoleToggle() {
  consoleCollapsed = !consoleCollapsed;
  const panel = document.getElementById('console-panel');
  const btn = document.getElementById('console-toggle-btn');
  if (consoleCollapsed) {
    panel.classList.add('collapsed');
    btn.textContent = '◀';
    btn.title = 'Expand Live Console';
  } else {
    panel.classList.remove('collapsed');
    panel.style.width = consoleWidth + 'px';
    btn.textContent = '▶';
    btn.title = 'Collapse Live Console';
  }
}

function consoleClear() {
  consoleLines = [];
  document.getElementById('console-body').innerHTML = '';
}

function cpSelectTab(el) {
  document.querySelectorAll('#console-header .fv-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  cpActiveTab = el.dataset.tab;

  const consoleBody = document.getElementById('console-body');
  const fileBody = document.getElementById('console-file-body');
  const consoleCtrl = document.getElementById('console-controls');
  const fileCtrl = document.getElementById('file-controls');
  const vsel = document.getElementById('fv-version-select');

  if (cpActiveTab === 'console') {
    consoleBody.style.display = '';
    fileBody.style.display = 'none';
    consoleCtrl.style.display = '';
    fileCtrl.style.display = 'none';
  } else {
    consoleBody.style.display = 'none';
    fileBody.style.display = '';
    consoleCtrl.style.display = 'none';
    fileCtrl.style.display = '';
    fvCurrentFile = cpActiveTab;
    vsel.style.display = cpActiveTab === 'artifact' ? 'inline-block' : 'none';
    fvRefresh();
  }
}

function consoleColorClass(line) {
  if (line.startsWith('  \u2192 ')) return 'cl-tool';
  if (line.startsWith('  Crav:')) return 'cl-crav';
  if (line.startsWith('  Judge:')) {
    if (/ PASS/.test(line)) return 'cl-pass';
    if (/ FAIL/.test(line)) return 'cl-fail';
    return 'cl-judge';
  }
  if (/ PASS/.test(line)) return 'cl-pass';
  if (/ FAIL/.test(line)) return 'cl-fail';
  if (/\[E\d+\] Complete:/.test(line) || /Complete:/.test(line)) return 'cl-epoch';
  if (/OOM|WARNING|WARN/.test(line)) return 'cl-warn';
  return 'cl-info';
}

function renderConsoleLine(line) {
  const cls = consoleColorClass(line);
  // Crav text lines: click-to-expand (first 100 chars collapsed)
  const CRAV_PREFIX = '  Crav: "';
  if (line.startsWith(CRAV_PREFIX)) {
    const inner = line.slice(CRAV_PREFIX.length, line.endsWith('"') ? -1 : undefined);
    if (inner.length > 100) {
      const short = escHtml(inner.slice(0, 100));
      const full  = escHtml(inner);
      return `<div class="cl-line cl-crav" onclick="cravToggle(this)">`
        + `<span class="crav-short">  Crav: &ldquo;${short}&hellip;<span class="crav-toggle">[+]</span>&rdquo;</span>`
        + `<span class="crav-full" style="display:none">  Crav: &ldquo;${full}&rdquo;</span>`
        + `</div>`;
    }
    return `<div class="cl-line cl-crav">  Crav: &ldquo;${escHtml(inner)}&rdquo;</div>`;
  }
  return `<div class="cl-line ${cls}">${escHtml(line)}</div>`;
}

function cravToggle(el) {
  const short = el.querySelector('.crav-short');
  const full  = el.querySelector('.crav-full');
  const expanded = full.style.display !== 'none';
  full.style.display  = expanded ? 'none' : '';
  short.style.display = expanded ? '' : 'none';
}

function consoleUpdate(lines) {
  if (!lines || !lines.length) return;
  if (lines.length === consoleLines.length) return;
  consoleLines = lines;
  const body = document.getElementById('console-body');

  // Preserve expanded crav toggle states before rebuild.
  const expandedSet = new Set();
  body.querySelectorAll('.cl-crav').forEach((el, i) => {
    const full = el.querySelector('.crav-full');
    if (full && full.style.display !== 'none') expandedSet.add(i);
  });

  body.innerHTML = lines.map(renderConsoleLine).join('');

  // Restore expanded states.
  if (expandedSet.size > 0) {
    body.querySelectorAll('.cl-crav').forEach((el, i) => {
      if (expandedSet.has(i)) {
        const short = el.querySelector('.crav-short');
        const full = el.querySelector('.crav-full');
        if (short && full) { short.style.display = 'none'; full.style.display = ''; }
      }
    });
  }

  const autoScroll = document.getElementById('console-autoscroll');
  if (autoScroll && autoScroll.checked) {
    body.scrollTop = body.scrollHeight;
  }
}

// ---- Console resize (drag left border) ----
(function() {
  const handle = document.getElementById('console-drag-handle');
  handle.addEventListener('mousedown', function(e) {
    e.preventDefault();
    const startX = e.clientX;
    const startWidth = consoleWidth;
    handle.classList.add('dragging');
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';

    function onMove(e) {
      const delta = startX - e.clientX;
      const newWidth = Math.max(200, Math.min(window.innerWidth * 0.8, startWidth + delta));
      consoleWidth = newWidth;
      document.getElementById('console-panel').style.width = newWidth + 'px';
    }
    function onUp() {
      handle.classList.remove('dragging');
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
})();

// ---- Console wrap toggle ----
document.getElementById('console-wrap').addEventListener('change', function() {
  const body = document.getElementById('console-body');
  if (this.checked) {
    body.style.whiteSpace = 'pre-wrap';
    body.style.wordBreak = 'break-word';
    body.style.overflowX = 'hidden';
  } else {
    body.style.whiteSpace = 'pre';
    body.style.wordBreak = '';
    body.style.overflowX = 'auto';
  }
});

// ============================================================
// File Viewer
// ============================================================
// fvSelectTab removed — replaced by cpSelectTab in console panel.

function fvFetch(url) {
  return fetch(url + '?_t=' + Date.now()).then(r => {
    if (!r.ok) return r.text().then(t => { throw new Error(t || r.statusText); });
    const hash = r.headers.get('X-Content-Hash') || '';
    return r.text().then(txt => ({ txt, hash }));
  });
}

function fvRefresh() {
  if (fvCurrentFile === 'artifact') {
    fvLoadArtifact();
  } else {
    fvFetch('/api/files/' + fvCurrentFile)
      .then(({txt, hash}) => {
        fvTrackVersion(fvCurrentFile, hash);
        fvDisplay(txt, fvCurrentFile);
      })
      .catch(e => fvDisplay('(waiting for server…)', 'txt'));
  }
}

function fvTrackVersion(filename, hash) {
  if (!hash) return;
  const prev = fvHashes[filename];
  if (prev === undefined) {
    // First load: initialise at v1.
    fvHashes[filename] = hash;
    fvVersions[filename] = 1;
  } else if (prev !== hash) {
    // Content changed — bump version.
    fvHashes[filename] = hash;
    fvVersions[filename] = (fvVersions[filename] || 1) + 1;
  }
  // Update the badge on the compress.py tab.
  if (filename === 'compress.py') {
    const badge = document.getElementById('fv-ver-compress');
    if (badge) badge.textContent = 'v' + (fvVersions[filename] || 1);
  }
}

function fvLoadArtifact() {
  const sel = document.getElementById('fv-version-select');
  const ver = sel.value;
  if (!ver) { fvDisplay('No artifacts exported yet.', 'txt'); return; }
  fvFetch('/api/artifacts/' + ver)
    .then(({txt}) => fvDisplay(txt, 'compress.py'))
    .catch(e => fvDisplay('Artifact not available: ' + e.message, 'txt'));
}

function fvViewArtifact(version) {
  // Switch to artifact tab in console panel and load that version.
  const artTab = document.querySelector('#console-header .fv-tab[data-tab="artifact"]');
  if (artTab) cpSelectTab(artTab);
  document.getElementById('fv-version-select').value = String(version);
  fvLoadArtifact();
}

function updateVersionSelect(versions) {
  const sel = document.getElementById('fv-version-select');
  const cur = sel.value;
  sel.innerHTML = '<option value="">— version —</option>';
  for (const v of [...versions].reverse()) {
    const opt = document.createElement('option');
    opt.value = v.version;
    opt.textContent = `v${String(v.version).padStart(4,'0')} · ep${v.epoch} · ${(v.mean_score*100).toFixed(1)}%`;
    sel.appendChild(opt);
  }
  if (cur) sel.value = cur;
}

function fvDisplay(content, filename) {
  const el = document.getElementById('fv-content');
  if (filename.endsWith('.md')) {
    el.innerHTML = renderMarkdown(content);
    el.style.whiteSpace = 'normal';
  } else if (filename.endsWith('.py')) {
    el.innerHTML = renderPython(content);
    el.style.whiteSpace = 'pre';
  } else {
    el.textContent = content;
    el.style.whiteSpace = 'pre';
  }
}

// Auto-refresh: refresh current file tab content (only when a file tab is active).
setInterval(() => {
  if (document.getElementById('fv-auto-refresh').checked && cpActiveTab !== 'console') fvRefresh();
}, 5000);

// Always poll compress.py hash so the version badge updates
// even when viewing another tab.
setInterval(() => {
  fetch('/api/files/compress.py?_t=' + Date.now())
    .then(r => { fvTrackVersion('compress.py', r.headers.get('X-Content-Hash') || ''); })
    .catch(() => {});
}, 4000);

// ============================================================
// Syntax Highlighting: Markdown
// ============================================================
function renderMarkdown(text) {
  const lines = text.split('\n');
  let html = '';
  let inCode = false;
  let codeLines = [];

  for (const rawLine of lines) {
    if (rawLine.startsWith('```')) {
      if (inCode) {
        html += `<span class="md-codeblock">${escHtml(codeLines.join('\n'))}</span>`;
        codeLines = []; inCode = false;
      } else { inCode = true; }
      continue;
    }
    if (inCode) { codeLines.push(rawLine); continue; }

    let line = rawLine;
    if (line.startsWith('### ')) { html += `<div class="md-h3">${escHtml(line.slice(4))}</div>\n`; continue; }
    if (line.startsWith('## '))  { html += `<div class="md-h2">${escHtml(line.slice(3))}</div>\n`; continue; }
    if (line.startsWith('# '))   { html += `<div class="md-h1">${escHtml(line.slice(2))}</div>\n`; continue; }

    // Inline: bold, code
    line = escHtml(line);
    line = line.replace(/\*\*(.+?)\*\*/g, '<span class="md-bold">$1</span>');
    line = line.replace(/`(.+?)`/g, '<span class="md-code">$1</span>');

    if (rawLine.match(/^\s*[-*] /)) {
      html += `<div class="md-li">${line.replace(/^\s*[-*] /, '')}</div>\n`;
    } else {
      html += `<div style="min-height:1.4em">${line}</div>\n`;
    }
  }
  if (inCode) html += `<span class="md-codeblock">${escHtml(codeLines.join('\n'))}</span>`;
  return html;
}

// ============================================================
// Syntax Highlighting: Python
// ============================================================
const PY_KEYWORDS = new Set(['def','class','return','import','from','if','elif','else','for','while','try','except','finally','with','as','pass','break','continue','raise','yield','lambda','and','or','not','in','is','None','True','False','self','cls','async','await','global','nonlocal','del','assert']);

function renderPython(code) {
  const lines = code.split('\n');
  let html = '';
  for (let i = 0; i < lines.length; i++) {
    const ln = `<span class="py-ln">${i+1}</span>`;
    html += ln + tokenizePyLine(lines[i]) + '\n';
  }
  return html;
}

function tokenizePyLine(line) {
  // Handle full-line comment
  const trimmed = line.trimStart();
  const indent = line.length - trimmed.length;
  const indentStr = escHtml(line.substring(0, indent));

  if (trimmed.startsWith('#')) {
    return indentStr + `<span class="py-cmt">${escHtml(trimmed)}</span>`;
  }

  // Decorator
  if (trimmed.startsWith('@')) {
    return indentStr + `<span class="py-deco">${escHtml(trimmed)}</span>`;
  }

  // Simple tokenizer: strings, comments, keywords, numbers
  let result = indentStr;
  let rest = trimmed;
  while (rest.length > 0) {
    // Comment
    const cIdx = rest.indexOf('#');
    // String (simplified: single/double quoted, no multi-line)
    const sqIdx = rest.indexOf("'");
    const dqIdx = rest.indexOf('"');

    // Find first string or comment
    let firstStr = -1;
    let strChar = '';
    if (sqIdx >= 0 && (dqIdx < 0 || sqIdx <= dqIdx)) { firstStr = sqIdx; strChar = "'"; }
    if (dqIdx >= 0 && (sqIdx < 0 || dqIdx < sqIdx))  { firstStr = dqIdx; strChar = '"'; }

    // Comment before string
    if (cIdx >= 0 && (firstStr < 0 || cIdx < firstStr)) {
      result += tokenizeKeywords(rest.substring(0, cIdx));
      result += `<span class="py-cmt">${escHtml(rest.substring(cIdx))}</span>`;
      break;
    }

    // String before comment
    if (firstStr >= 0) {
      result += tokenizeKeywords(rest.substring(0, firstStr));
      rest = rest.substring(firstStr);
      // Find matching close quote (simple, no escape handling)
      const closeIdx = rest.indexOf(strChar, 1);
      if (closeIdx < 0) {
        result += `<span class="py-str">${escHtml(rest)}</span>`;
        rest = '';
      } else {
        result += `<span class="py-str">${escHtml(rest.substring(0, closeIdx + 1))}</span>`;
        rest = rest.substring(closeIdx + 1);
      }
      continue;
    }

    // No special chars
    result += tokenizeKeywords(rest);
    break;
  }
  return result;
}

function tokenizeKeywords(text) {
  if (!text) return '';
  // Split by word boundaries
  return escHtml(text).replace(/\b([a-zA-Z_]\w*)\b/g, (m) => {
    if (PY_KEYWORDS.has(m)) return `<span class="py-kw">${m}</span>`;
    // Check if followed by ( = function call
    return m;
  }).replace(/\b(\d+\.?\d*)\b/g, `<span class="py-num">$1</span>`);
}

// ============================================================
// Utilities
// ============================================================
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function fmt(n) { return Number(n).toLocaleString(); }
function pct(v) { return (v * 100).toFixed(1) + '%'; }
function pctColor(v) { return v >= 85 ? 'good' : v >= 60 ? 'warn' : 'bad'; }
function setStatVal(id, txt, cls) {
  const el = document.getElementById(id);
  el.textContent = txt;
  if (cls) el.className = 'stat-val ' + cls;
}

// ============================================================
// Init
// ============================================================
connect();

// Initial file load deferred — starts on Console tab.

// Add info log entry on connect
setTimeout(() => addLog(new Date().toLocaleTimeString(), '—', 'log-info', 'Dashboard connected — watching for epochs…'), 500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard Server
# ---------------------------------------------------------------------------

class DashboardServer:
    """Serves real-time metrics over WebSocket on the configured port."""

    def __init__(self, config: dict, run_dir: str):
        self.config = config
        self.run_dir = run_dir
        self.storage = MetricsStorage(run_dir)
        self.collector = MetricsCollector(self.storage, config)
        self._control_path = os.path.join(run_dir, "control.json")
        self.app = self._create_app()

    # ------------------------------------------------------------------
    # App factory
    # ------------------------------------------------------------------

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="CravingMind Dashboard")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(DASHBOARD_HTML)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            interval = float(
                self.config.get("dashboard", {}).get("update_interval_seconds", 2)
            )
            try:
                while True:
                    state = self.collector.get_dashboard_state()
                    state["control"] = self._read_control()
                    await websocket.send_json(state)
                    await asyncio.sleep(interval)
            except WebSocketDisconnect:
                pass

        @app.get("/api/console")
        async def console_lines():
            return JSONResponse(self.storage.get_console_lines(limit=200))

        @app.get("/api/epochs")
        async def epochs():
            return JSONResponse(self.storage.get_epoch_history())

        @app.get("/api/tasks/{epoch}")
        async def tasks(epoch: int):
            return JSONResponse(self.storage.get_task_history(epoch))

        @app.get("/api/artifacts")
        async def artifacts():
            return JSONResponse(self.storage.get_artifact_history())

        @app.get("/api/files/{filename}")
        async def file_content(filename: str):
            """Return contents of bible.md, graveyard.md, or compress.py.

            Returns an X-Content-Hash header (first 12 hex of SHA-256)
            so the dashboard can detect when a file actually changes and
            bump its visual version counter.
            """
            import hashlib
            allowed = {"bible.md", "graveyard.md", "compress.py"}
            if filename not in allowed:
                return PlainTextResponse("File not found.", status_code=404)
            # Look in agent_workspace subdir
            agent_dir = os.path.join(self.run_dir, "agent_workspace")
            path = os.path.join(agent_dir, filename)
            headers = {"Cache-Control": "no-store"}
            if not os.path.exists(path):
                body = f"# {filename}\n\n(not yet created)"
                headers["X-Content-Hash"] = hashlib.sha256(body.encode()).hexdigest()[:12]
                return PlainTextResponse(
                    body,
                    status_code=200,
                    headers=headers,
                )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                headers["X-Content-Hash"] = hashlib.sha256(content.encode()).hexdigest()[:12]
                return PlainTextResponse(content, headers=headers)
            except OSError as exc:
                return PlainTextResponse(f"Error reading file: {exc}", status_code=500)

        @app.get("/api/artifacts/{version}")
        async def artifact_version(version: int):
            """Return contents of a specific artifact version."""
            history = self.storage.get_artifact_history()
            entry = next((a for a in history if a.get("version") == version), None)
            if not entry:
                return PlainTextResponse("Artifact version not found.", status_code=404)
            filepath = entry.get("filepath", "")
            if not os.path.exists(filepath):
                return PlainTextResponse("Artifact file missing from disk.", status_code=404)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return PlainTextResponse(f.read())
            except OSError as exc:
                return PlainTextResponse(f"Error reading artifact: {exc}", status_code=500)

        @app.post("/api/control")
        async def control(request: Request):
            """Set run control signals (pause/resume/stop)."""
            import json as _json
            body = await request.json()
            action = body.get("action", "")

            # Read current state.
            ctrl = self._read_control()

            if action == "pause":
                ctrl["paused"] = True
            elif action == "resume":
                ctrl["paused"] = False
            elif action == "stop":
                ctrl["stopped"] = True
            else:
                return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

            self._write_control(ctrl)
            return JSONResponse({"ok": True, **ctrl})

        @app.get("/api/control")
        async def control_state():
            return JSONResponse(self._read_control())

        return app

    def _read_control(self) -> dict:
        """Read control.json; return defaults if missing."""
        try:
            with open(self._control_path, "r", encoding="utf-8") as f:
                import json as _json
                return _json.load(f)
        except (OSError, ValueError):
            return {"paused": False, "stopped": False}

    def _write_control(self, ctrl: dict) -> None:
        """Write control.json atomically."""
        import json as _json
        try:
            with open(self._control_path, "w", encoding="utf-8") as f:
                _json.dump(ctrl, f)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self, host: str = "0.0.0.0", port: int | None = None):
        import uvicorn
        port = port or int(self.config.get("dashboard", {}).get("port", 8080))
        uvicorn.run(self.app, host=host, port=port, log_level="warning")
