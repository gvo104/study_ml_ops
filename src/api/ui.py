from __future__ import annotations


BASE_STYLES = """
<style>
  :root {
    color-scheme: light;
    --bg: #f3f6fb;
    --surface: #ffffff;
    --surface-muted: #eef3f8;
    --border: #d6dfeb;
    --text: #18212f;
    --muted: #5f6f85;
    --accent: #2563eb;
    --accent-strong: #1d4ed8;
    --success: #157347;
    --warn: #b45309;
    --danger: #b42318;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
  }

  .shell {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 240px minmax(0, 1fr);
  }

  .nav {
    background: #101828;
    color: #f8fafc;
    padding: 24px 18px;
  }

  .nav h1 {
    margin: 0 0 24px;
    font-size: 20px;
    line-height: 1.3;
  }

  .nav a {
    display: block;
    margin-bottom: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    color: #cbd5e1;
    text-decoration: none;
    background: transparent;
  }

  .nav a.active {
    background: rgba(37, 99, 235, 0.18);
    color: #ffffff;
  }

  .main {
    padding: 28px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: start;
    margin-bottom: 24px;
  }

  .header h2 {
    margin: 0;
    font-size: 28px;
  }

  .header p {
    margin: 8px 0 0;
    color: var(--muted);
  }

  .grid {
    display: grid;
    gap: 16px;
  }

  .grid.cols-2 {
    grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px;
  }

  .panel h3 {
    margin: 0 0 12px;
    font-size: 18px;
  }

  .stack {
    display: grid;
    gap: 12px;
  }

  textarea {
    width: 100%;
    min-height: 180px;
    resize: vertical;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    font: inherit;
    color: var(--text);
    background: #fcfdff;
  }

  button {
    border: 0;
    border-radius: 8px;
    padding: 11px 16px;
    font: inherit;
    cursor: pointer;
    background: var(--accent);
    color: #fff;
  }

  button.secondary {
    background: #dbe8ff;
    color: #163b87;
  }

  button:disabled {
    opacity: 0.6;
    cursor: progress;
  }

  .result {
    padding: 14px;
    border-radius: 8px;
    background: var(--surface-muted);
  }

  .metric {
    display: grid;
    gap: 6px;
  }

  .metric strong {
    font-size: 22px;
  }

  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 6px 10px;
    border-radius: 999px;
    background: #dbe8ff;
    color: #163b87;
    font-size: 13px;
  }

  .badge.warn {
    background: #fff1d6;
    color: var(--warn);
  }

  .badge.danger {
    background: #fee4e2;
    color: var(--danger);
  }

  .notice {
    padding: 14px 16px;
    border-radius: 8px;
    border: 1px solid #f2d28a;
    background: #fff8e8;
    color: #8a5a00;
  }

  .notice.high {
    border-color: #f0a4a1;
    background: #fff0ef;
    color: var(--danger);
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 12px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
    font-size: 14px;
  }

  th {
    color: var(--muted);
    font-weight: 600;
  }

  .muted {
    color: var(--muted);
  }

  .mono {
    font-family: Consolas, Menlo, Monaco, monospace;
    font-size: 13px;
  }

  .inline {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
  }

  @media (max-width: 980px) {
    .shell {
      grid-template-columns: 1fr;
    }

    .nav {
      padding-bottom: 10px;
    }

    .grid.cols-2 {
      grid-template-columns: 1fr;
    }
  }
</style>
"""


def _layout(title: str, active: str, content: str, script: str) -> str:
    inference_class = "active" if active == "inference" else ""
    experiments_class = "active" if active == "experiments" else ""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  {BASE_STYLES}
</head>
<body>
  <div class="shell">
    <aside class="nav">
      <h1>MLOps Control Center</h1>
      <a class="{inference_class}" href="/">Inference</a>
      <a class="{experiments_class}" href="/experiments">Experiments</a>
    </aside>
    <main class="main">
      {content}
    </main>
  </div>
  <script>
    const formatTs = (value) => value ? new Date(value).toLocaleString() : "-";
    const formatMlflowTs = (value) => value ? new Date(value).toLocaleString() : "-";
    const escapeHtml = (value) => String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
    {script}
  </script>
</body>
</html>
"""


def render_inference_page() -> str:
    content = """
    <section class="header">
      <div>
        <h2>Inference Console</h2>
        <p>Run single-text predictions, review recent traffic, and watch drift signals.</p>
      </div>
    </section>

    <section class="grid cols-2">
      <div class="panel stack">
        <h3>Input</h3>
        <textarea id="text-input" placeholder="Paste text for classification"></textarea>
        <div class="inline">
          <button id="predict-btn">Run inference</button>
          <span id="predict-status" class="muted"></span>
        </div>
      </div>

      <div class="panel stack">
        <h3>Result</h3>
        <div id="prediction-result" class="result">
          <div class="muted">No prediction yet.</div>
        </div>
      </div>
    </section>

    <section class="grid" style="margin-top: 16px;">
      <div class="panel stack">
        <h3>Drift notifications</h3>
        <div id="drift-list" class="stack">
          <div class="muted">No active drift notifications.</div>
        </div>
      </div>

      <div class="panel">
        <h3>Latest predictions</h3>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Text</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Anomaly flags</th>
            </tr>
          </thead>
          <tbody id="predictions-body">
            <tr><td colspan="5" class="muted">No requests yet.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
    """

    script = """
    const textInput = document.getElementById("text-input");
    const predictBtn = document.getElementById("predict-btn");
    const predictStatus = document.getElementById("predict-status");
    const predictionResult = document.getElementById("prediction-result");
    const predictionsBody = document.getElementById("predictions-body");
    const driftList = document.getElementById("drift-list");

    const renderFlags = (flags) => {
      if (!flags.length) {
        return '<span class="badge">normal</span>';
      }
      return flags.map((flag) => `<span class="badge warn">${escapeHtml(flag)}</span>`).join("");
    };

    const renderSummary = (payload) => {
      const notifications = payload.drift_notifications || [];
      if (!notifications.length) {
        driftList.innerHTML = '<div class="muted">No active drift notifications.</div>';
      } else {
        driftList.innerHTML = notifications.map((item) => `
          <div class="notice ${item.level === "high" ? "high" : ""}">
            <strong>${item.level.toUpperCase()}</strong><br />
            ${escapeHtml(item.message)}
          </div>
        `).join("");
      }

      const rows = payload.recent_predictions || [];
      if (!rows.length) {
        predictionsBody.innerHTML = '<tr><td colspan="5" class="muted">No requests yet.</td></tr>';
        return;
      }

      predictionsBody.innerHTML = rows.map((row) => `
        <tr>
          <td>${formatTs(row.timestamp)}</td>
          <td>${escapeHtml(row.text_preview)}</td>
          <td>${escapeHtml(row.prediction)}</td>
          <td>${(row.confidence * 100).toFixed(1)}%</td>
          <td><div class="badges">${renderFlags(row.anomaly_flags)}</div></td>
        </tr>
      `).join("");
    };

    const loadDashboard = async () => {
      const response = await fetch("/api/dashboard");
      const payload = await response.json();
      renderSummary(payload);
    };

    predictBtn.addEventListener("click", async () => {
      const text = textInput.value.trim();
      if (!text) {
        predictStatus.textContent = "Enter some text first.";
        return;
      }

      predictBtn.disabled = true;
      predictStatus.textContent = "Running inference...";

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const payload = await response.json();
      predictBtn.disabled = false;

      if (!response.ok) {
        predictStatus.textContent = payload.detail || "Request failed.";
        return;
      }

      predictStatus.textContent = "Done.";
      const probabilities = Object.entries(payload.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([label, value]) => `<div>${escapeHtml(label)}: ${(value * 100).toFixed(1)}%</div>`)
        .join("");

      predictionResult.innerHTML = `
        <div class="metric">
          <span class="muted">Predicted label</span>
          <strong>${escapeHtml(payload.prediction)}</strong>
        </div>
        <div class="metric">
          <span class="muted">Confidence</span>
          <strong>${(payload.confidence * 100).toFixed(1)}%</strong>
        </div>
        <div class="stack">
          <span class="muted">Class probabilities</span>
          ${probabilities}
        </div>
        <div class="badges" style="margin-top: 10px;">
          ${renderFlags(payload.anomaly_flags || [])}
        </div>
      `;

      await loadDashboard();
    });

    loadDashboard();
    """

    return _layout("Inference Console", "inference", content, script)


def render_experiments_page() -> str:
    content = """
    <section class="header">
      <div>
        <h2>Experiments & Retraining</h2>
        <p>Track candidate configs, current production metrics, and retraining progress.</p>
      </div>
      <button id="retrain-btn">Start retraining</button>
    </section>

    <section class="grid cols-2">
      <div class="panel stack">
        <h3>Production model</h3>
        <div id="active-model" class="stack muted">Loading model metadata...</div>
      </div>
      <div class="panel stack">
        <h3>Retraining status</h3>
        <div id="retraining-status" class="stack muted">Loading retraining status...</div>
      </div>
    </section>

    <section class="grid" style="margin-top: 16px;">
      <div class="panel stack">
        <h3>Drift notifications</h3>
        <div id="experiment-drift" class="stack">
          <div class="muted">No active drift notifications.</div>
        </div>
      </div>

      <div class="panel">
        <h3>MLflow runs</h3>
        <table>
          <thead>
            <tr>
              <th>Run name</th>
              <th>Status</th>
              <th>Started</th>
              <th>Metrics</th>
              <th>Params</th>
            </tr>
          </thead>
          <tbody id="runs-body">
            <tr><td colspan="5" class="muted">Loading MLflow runs...</td></tr>
          </tbody>
        </table>
      </div>

      <div class="panel">
        <h3>Experiment configs</h3>
        <table>
          <thead>
            <tr>
              <th>Run name</th>
              <th>Config</th>
              <th>Model</th>
              <th>Features</th>
            </tr>
          </thead>
          <tbody id="configs-body">
            <tr><td colspan="4" class="muted">Loading configs...</td></tr>
          </tbody>
        </table>
      </div>
    </section>
    """

    script = """
    const retrainBtn = document.getElementById("retrain-btn");
    const activeModel = document.getElementById("active-model");
    const retrainingStatus = document.getElementById("retraining-status");
    const runsBody = document.getElementById("runs-body");
    const configsBody = document.getElementById("configs-body");
    const experimentDrift = document.getElementById("experiment-drift");

    const renderNotifications = (items) => {
      if (!items.length) {
        experimentDrift.innerHTML = '<div class="muted">No active drift notifications.</div>';
        return;
      }

      experimentDrift.innerHTML = items.map((item) => `
        <div class="notice ${item.level === "high" ? "high" : ""}">
          <strong>${item.level.toUpperCase()}</strong><br />
          ${escapeHtml(item.message)}
        </div>
      `).join("");
    };

    const formatMetric = (metrics, name) => {
      const value = metrics?.[name];
      if (value === undefined || value === null) {
        return "-";
      }
      return (Number(value) * 100).toFixed(1) + "%";
    };

    const renderRunParams = (params) => {
      const keys = ["model.name", "features.tfidf_max_features", "features.svd_components", "training.oversample"];
      const visible = keys
        .filter((key) => params?.[key] !== undefined)
        .map((key) => `${escapeHtml(key)}=${escapeHtml(params[key])}`);
      return visible.length ? visible.join("<br />") : "-";
    };

    const loadExperiments = async () => {
      const response = await fetch("/api/experiments");
      const payload = await response.json();

      const modelStatus = payload.model_loaded ? "loaded" : "not loaded";
      activeModel.innerHTML = `
        <div><span class="muted">Model status</span><br /><strong>${modelStatus}</strong></div>
        <div><span class="muted">Feature schema</span><br /><strong>${escapeHtml(payload.active_model.feature_schema_version || "-")}</strong></div>
        <div><span class="muted">Accuracy</span><br /><strong>${payload.active_model.accuracy ? (payload.active_model.accuracy * 100).toFixed(1) + "%" : "-"}</strong></div>
        <div><span class="muted">Macro F1</span><br /><strong>${payload.active_model.macro_f1 ? (payload.active_model.macro_f1 * 100).toFixed(1) + "%" : "-"}</strong></div>
        <div><span class="muted">Created</span><br /><strong>${escapeHtml(payload.active_model.created_at || "-")}</strong></div>
        <div><span class="muted">MLflow</span><br /><strong>${payload.tracking.connected ? "connected" : "offline"}</strong></div>
        <div><span class="muted">Tracking URI</span><br /><span class="mono">${escapeHtml(payload.tracking.uri || "-")}</span></div>
      `;
      if (payload.startup_error) {
        activeModel.innerHTML += `
          <div class="notice high">
            <strong>Startup note</strong><br />
            ${escapeHtml(payload.startup_error)}
          </div>
        `;
      }
      if (payload.tracking.error) {
        activeModel.innerHTML += `
          <div class="notice high">
            <strong>MLflow note</strong><br />
            ${escapeHtml(payload.tracking.error)}
          </div>
        `;
      }

      const retraining = payload.retraining;
      retrainingStatus.innerHTML = `
        <div><span class="muted">State</span><br /><strong>${retraining.state}</strong></div>
        <div><span class="muted">Message</span><br /><span>${escapeHtml(retraining.message)}</span></div>
        <div><span class="muted">Started</span><br /><span>${formatTs(retraining.started_at)}</span></div>
        <div><span class="muted">Completed</span><br /><span>${formatTs(retraining.completed_at)}</span></div>
        <div><span class="muted">Last run</span><br /><span>${escapeHtml(retraining.last_run_name || "-")}</span></div>
      `;
      retrainBtn.disabled = retraining.state === "running";

      renderNotifications(payload.drift_notifications || []);

      const runs = payload.mlflow_runs || [];
      if (!runs.length) {
        runsBody.innerHTML = `<tr><td colspan="5" class="muted">${payload.tracking.connected ? "No MLflow runs yet." : "MLflow is offline or unreachable."}</td></tr>`;
      } else {
        runsBody.innerHTML = runs.map((row) => `
          <tr>
            <td>
              <strong>${escapeHtml(row.run_name)}</strong><br />
              <span class="muted mono">${escapeHtml(row.run_id)}</span>
            </td>
            <td>${escapeHtml(row.status)}</td>
            <td>${formatMlflowTs(row.start_time)}</td>
            <td>accuracy=${formatMetric(row.metrics, "accuracy")}<br />macro_f1=${formatMetric(row.metrics, "macro_f1")}</td>
            <td>${renderRunParams(row.params)}</td>
          </tr>
        `).join("");
      }

      const configs = payload.experiment_configs || [];
      configsBody.innerHTML = configs.map((row) => `
        <tr>
          <td>${escapeHtml(row.name)}</td>
          <td class="mono">${escapeHtml(row.config_path)}</td>
          <td>${escapeHtml(row.model)}</td>
          <td>tfidf=${escapeHtml(row.tfidf_max_features)}, svd=${escapeHtml(row.svd_components)}, oversample=${escapeHtml(row.oversample)}</td>
        </tr>
      `).join("");
    };

    retrainBtn.addEventListener("click", async () => {
      retrainBtn.disabled = true;
      const response = await fetch("/api/retrain", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) {
        alert(payload.detail || "Could not start retraining.");
      }
      await loadExperiments();
    });

    loadExperiments();
    setInterval(loadExperiments, 8000);
    """

    return _layout("Experiments", "experiments", content, script)
