const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

const fields = [
  ["duration","number","connection"],
  ["protocol_type","select","connection",["tcp","udp","icmp"]],
  ["service","text","connection"],
  ["flag","select","connection",["SF","S0","REJ","RSTR","RSTO","S1","S2","S3","OTH"]],
  ["src_bytes","number","connection"],
  ["dst_bytes","number","connection"],
  ["land","number","connection"],
  ["wrong_fragment","number","connection"],
  ["urgent","number","connection"],
  ["hot","number","content"],
  ["num_failed_logins","number","content"],
  ["logged_in","number","content"],
  ["num_compromised","number","content"],
  ["root_shell","number","content"],
  ["su_attempted","number","content"],
  ["num_root","number","content"],
  ["num_file_creations","number","content"],
  ["num_shells","number","content"],
  ["num_access_files","number","content"],
  ["num_outbound_cmds","number","content"],
  ["is_host_login","number","content"],
  ["is_guest_login","number","content"],
  ["count","number","traffic"],
  ["srv_count","number","traffic"],
  ["serror_rate","number","traffic",null,0,1],
  ["srv_serror_rate","number","traffic",null,0,1],
  ["rerror_rate","number","traffic",null,0,1],
  ["srv_rerror_rate","number","traffic",null,0,1],
  ["same_srv_rate","number","traffic",null,0,1],
  ["diff_srv_rate","number","traffic",null,0,1],
  ["srv_diff_host_rate","number","traffic",null,0,1],
  ["dst_host_count","number","host"],
  ["dst_host_srv_count","number","host"],
  ["dst_host_same_srv_rate","number","host",null,0,1],
  ["dst_host_diff_srv_rate","number","host",null,0,1],
  ["dst_host_same_src_port_rate","number","host",null,0,1],
  ["dst_host_srv_diff_host_rate","number","host",null,0,1],
  ["dst_host_serror_rate","number","host",null,0,1],
  ["dst_host_srv_serror_rate","number","host",null,0,1],
  ["dst_host_rerror_rate","number","host",null,0,1],
  ["dst_host_srv_rerror_rate","number","host",null,0,1]
];

// These are real records from the bundled NSL-KDD training set.
const normal = [0,"tcp","ftp_data","SF",491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,0,1,0,0,150,25,0.17,0.03,0.17,0,0,0,0.05,0];
const attack = [0,"tcp","private","S0",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1,1,0,0,0.05,0.07,0,255,26,0.10,0.05,0,0,1,1,0,0];

const state = {
  metadata:null,
  history:JSON.parse(localStorage.getItem("cypherx_history") || "[]"),
  records:[normal.slice()],
  activeRecord:0,
};

const $ = (id) => document.getElementById(id);
const pct = (x) => `${(Number(x) * 100).toFixed(2)}%`;

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(`${API_BASE}${path}`, {...options, signal: controller.signal});
    const text = await response.text();
    let data = {};
    try { data = text ? JSON.parse(text) : {}; } catch { data = {detail:text}; }
    if (!response.ok) {
      const detail = data?.detail;
      let message = `HTTP ${response.status}`;
      if (typeof detail === "string" && detail.trim()) message = detail;
      else if (Array.isArray(detail)) message = detail.map(item => {
        const loc = Array.isArray(item?.loc) ? item.loc.join(" → ") : "Request";
        return `${loc}: ${item?.msg || "Invalid value"}`;
      }).join(" | ");
      else if (detail && typeof detail === "object") message = JSON.stringify(detail);
      throw new Error(message);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The FastAPI server did not respond within 8 seconds.");
    throw error;
  } finally { clearTimeout(timer); }
}

function setApi(online, detail="") {
  $("apiDot").style.background = online ? "var(--green)" : "var(--red)";
  $("apiText").textContent = online ? "API online" : "API offline";
  $("apiDetail").textContent = detail || (online ? "FastAPI + trained model ready" : "Start the CypherX server");
  $("serverDot").style.background = online ? "var(--green)" : "var(--red)";
  $("serverText").textContent = online ? "Model online" : "Model offline";
}

function showToast(title, text) {
  $("toastTitle").textContent = title; $("toastText").textContent = text; $("toast").classList.remove("hidden");
  clearTimeout(showToast.timer); showToast.timer = setTimeout(() => $("toast").classList.add("hidden"), 3000);
}

function showError(message) { $("errorText").textContent = message; $("errorBox").classList.remove("hidden"); }
function hideError() { $("errorBox").classList.add("hidden"); }

function valuesToRecord(values) {
  return values.slice();
}

function recordObject(values) {
  const payload = {};
  fields.forEach((f, i) => {
    payload[f[0]] = f[1] === "select" || f[1] === "text" ? values[i] : Number(values[i]);
  });
  return payload;
}

function recordFromForm() {
  const current = state.records[state.activeRecord].slice();
  document.querySelectorAll("#predictionForm [name]").forEach(el => {
    const index = fields.findIndex(f => f[0] === el.name);
    if (index >= 0) current[index] = (el.tagName === "SELECT" || el.type === "text") ? el.value : Number(el.value);
  });
  state.records[state.activeRecord] = current;
  return recordObject(current);
}

function renderRecordTabs() {
  const tabs = $("recordTabs");
  tabs.innerHTML = state.records.map((_, i) => `<button type="button" class="record-tab ${i===state.activeRecord?"active":""}" data-record="${i}"><span>${i+1}</span> Connection ${i+1}${i===state.activeRecord?'<b>●</b>':''}</button>`).join("");
  tabs.querySelectorAll("[data-record]").forEach(btn => btn.addEventListener("click", () => {
    recordFromForm();
    state.activeRecord = Number(btn.dataset.record);
    renderInputs(state.records[state.activeRecord]);
  }));
  $("recordCount").textContent = `${state.records.length} / 10 connections`;
  $("activeRecordTitle").textContent = `Connection ${state.activeRecord + 1}`;
  $("activeRecordNumber").textContent = state.activeRecord + 1;
}

function renderInputs(values = state.records[state.activeRecord] || normal) {
  const groups = {connection:"connectionGrid",content:"contentGrid",traffic:"trafficGrid",host:"hostGrid"};
  Object.values(groups).forEach(id => $(id).innerHTML = "");
  fields.forEach((f, i) => {
    const wrap = document.createElement("div"); wrap.className = "feature";
    const label = document.createElement("label"); label.textContent = f[0];
    let input;
    if (f[1] === "select") {
      input = document.createElement("select");
      f[3].forEach(value => { const opt=document.createElement("option"); opt.value=value; opt.textContent=value; input.appendChild(opt); });
    } else {
      input=document.createElement("input"); input.type=f[1] === "text" ? "text" : "number";
      if(f[1] === "number") {
        input.step="any";
        if(f[4]!==undefined && f[4]!==null) input.min=f[4];
        if(f[5]!==undefined && f[5]!==null) input.max=f[5];
      }
    }
    input.name=f[0]; input.value=values[i]; input.required=true;
    wrap.append(label,input); $(groups[f[2]]).appendChild(wrap);
  });
  renderRecordTabs();
}

function addRecord(values = null) {
  if (state.records.length >= 10) {
    showToast("Queue limit", "You can classify up to 10 connections at once.");
    return;
  }
  recordFromForm();
  state.records.push(values ? values.slice() : state.records[state.records.length - 1].slice());
  state.activeRecord = state.records.length - 1;
  renderInputs(state.records[state.activeRecord]);
  showSection("detection");
}

function resetRecords() {
  state.records = [normal.slice()];
  state.activeRecord = 0;
  renderInputs(normal);
  $("result").classList.add("hidden"); $("resultEmpty").classList.remove("hidden");
  $("batchResults").classList.add("hidden");
}

function formPayload() { return recordFromForm(); }

async function loadMetadata() {
  try {
    const [health, metadata] = await Promise.all([api("/health"), api("/model-info")]);
    state.metadata=metadata; setApi(true, `${health.features} features • ${health.model}`);
    $("dashModel").textContent=metadata.model;
    $("dashAccuracy").textContent=pct(metadata.metrics.accuracy);
    $("dashPrecision").textContent=pct(metadata.metrics.precision);
    $("dashF1").textContent=pct(metadata.metrics.f1);
    $("selectedModelText").textContent=metadata.model;
    $("normalCount").textContent=metadata.test_distribution?.Normal ?? "—";
    $("dosCount").textContent=metadata.test_distribution?.DoS ?? "—";
    $("probeCount").textContent=metadata.test_distribution?.Probe ?? "—";
    $("rareCount").textContent=(metadata.test_distribution?.R2L ?? 0)+(metadata.test_distribution?.U2R ?? 0);
    renderAnalytics(); renderPerformance(); renderHistory();
  } catch (error) {
    setApi(false, error.message);
  }
}

async function predict() {
  hideError();
  const payload=formPayload();
  const btn=$("predictBtn");
  const original=btn.innerHTML; btn.disabled=true; btn.innerHTML="<span>Classifying…</span><b>⟳</b>";
  const started=performance.now();
  try {
    const result=await api("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    const elapsed=Math.round(performance.now()-started);
    renderResult(result);
    addHistory(result, `Connection ${state.activeRecord + 1}`);
    showToast("Detection complete",`${result.prediction} • ${pct(result.confidence)} confidence • ${elapsed} ms`);
  } catch(error) {
    showError(error.message);
    showToast("Prediction failed", error.message);
  } finally { btn.disabled=false; btn.innerHTML=original; }
}

async function predictAll() {
  hideError();
  recordFromForm();
  const btn=$("predictAll");
  const original=btn.innerHTML; btn.disabled=true; btn.innerHTML="<span>Classifying all…</span><b>⟳</b>";
  const started=performance.now();
  try {
    const result=await api("/predict-batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(state.records.map(recordObject))});
    const elapsed=Math.round(performance.now()-started);
    renderBatchResults(result.results);
    result.results.forEach((r,i)=>addHistory(r, `Connection ${i+1}`));
    showToast("Batch detection complete",`${result.count} connections classified in ${elapsed} ms`);
  } catch(error) {
    showError(error.message);
    showToast("Batch prediction failed", error.message);
  } finally { btn.disabled=false; btn.innerHTML=original; }
}

function addHistory(result, source="Connection") {
  state.history.unshift({time:new Date().toLocaleString(),source,prediction:result.prediction,category:result.attack_category,confidence:result.confidence,model:result.model});
  state.history=state.history.slice(0,100); localStorage.setItem("cypherx_history",JSON.stringify(state.history)); renderHistory();
}

function renderBatchResults(results) {
  $("resultEmpty").classList.add("hidden"); $("result").classList.add("hidden"); $("batchResults").classList.remove("hidden");
  const attacks=results.filter(r=>r.prediction==="Attack").length;
  $("batchSummaryText").textContent=`${results.length} classified • ${attacks} attack • ${results.length-attacks} normal`;
  $("batchResultList").innerHTML=results.map(r=>`<button class="batch-result-row" type="button" data-result-index="${r.index-1}"><span class="batch-index">${r.index}</span><span><strong>Connection ${r.index}</strong><small>${r.model}</small></span><b class="status ${r.prediction.toLowerCase()}">${r.prediction}</b><span>${pct(r.confidence)}</span><i>→</i></button>`).join("");
  $("batchResultList").querySelectorAll("[data-result-index]").forEach(btn=>btn.addEventListener("click",()=>{state.activeRecord=Number(btn.dataset.resultIndex); renderInputs(state.records[state.activeRecord]); renderResult(results[state.activeRecord]);}));
}

function renderResult(r) {
  $("resultEmpty").classList.add("hidden"); $("result").classList.remove("hidden");
  const attackResult=r.prediction === "Attack";
  const badge=$("resultBadge"); badge.textContent=r.prediction.toUpperCase(); badge.className=`result-badge ${attackResult ? "attack" : "normal"}`;
  $("resultTitle").textContent=attackResult ? "Suspicious traffic detected" : "Traffic appears normal";
  $("resultNote").textContent=attackResult ? "The model classified this connection as attack traffic based on its learned NSL-KDD patterns." : "The model classified this connection as normal traffic based on its learned NSL-KDD patterns.";
  $("confidence").textContent=pct(r.confidence); $("confidenceBar").style.width=pct(r.confidence);
  $("attackType").textContent=r.attack_category; $("resultModel").textContent=r.model; $("resultFields").textContent=`${r.features_received} / 41`; $("resultTime").textContent=new Date().toLocaleTimeString();
  $("resultAdvice").textContent=attackResult ? "Demo note: live inference is binary Normal vs Attack. The Analytics section contains category-level evaluation for DoS, Probe, R2L and U2R." : "Demo note: this is a model prediction, not a guarantee that a real-world connection is safe.";
}

function renderPerformance() {
  const rows=state.metadata?.model_comparison || [];
  $("modelTable").innerHTML=`<table class="table"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${rows.map(x=>`<tr class="${x.model===state.metadata.model?"best-row":""}"><td><strong>${x.model}</strong>${x.model===state.metadata.model?'<span class="model-tag">LIVE</span>':''}</td><td>${pct(x.accuracy)}</td><td>${pct(x.precision)}</td><td>${pct(x.recall)}</td><td><strong>${pct(x.f1)}</strong></td></tr>`).join("")}</tbody></table>`;
}

function renderAnalytics() {
  const cm=state.metadata?.confusion_matrix || [[0,0],[0,0]];
  $("confusion").innerHTML=`<div class="axis"></div><div class="axis">Pred Normal</div><div class="axis">Pred Attack</div><div class="axis">Actual Normal</div><div class="good">${cm[0][0]}</div><div class="bad">${cm[0][1]}</div><div class="axis">Actual Attack</div><div class="bad">${cm[1][0]}</div><div class="good">${cm[1][1]}</div>`;
  const fi=state.metadata?.feature_importance || []; const max=Math.max(...fi.map(x=>x.importance),1);
  $("features").innerHTML=fi.slice(0,10).map(x=>`<div class="bar-row"><div class="bar-top"><strong>${x.feature}</strong><span>${x.importance.toFixed(4)}</span></div><div class="bar-track"><i style="width:${x.importance/max*100}%"></i></div></div>`).join("");
  const cats=state.metadata?.attack_category_analysis || [];
  $("categoryBars").innerHTML=cats.map(x=>`<div class="category-card"><strong>${x.category}</strong><span>${pct(x.attack_detection_recall)}</span><small>${x.samples.toLocaleString()} samples</small><div class="category-track"><i style="width:${x.attack_detection_recall*100}%"></i></div></div>`).join("");
}

function renderHistory() {
  const query=($("historySearch")?.value || "").toLowerCase();
  const rows=state.history.filter(x=>JSON.stringify(x).toLowerCase().includes(query));
  $("historyCount").textContent=`${state.history.length} detection${state.history.length===1?"":"s"}`;
  $("historyTable").innerHTML=rows.length?`<table class="table"><thead><tr><th>Time</th><th>Source</th><th>Prediction</th><th>Category</th><th>Confidence</th><th>Model</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${x.time}</td><td>${x.source || "Connection"}</td><td class="status ${x.prediction.toLowerCase()}">${x.prediction}</td><td>${x.category}</td><td>${pct(x.confidence)}</td><td>${x.model}</td></tr>`).join("")}</tbody></table>`:`<div class="empty-table">No detections match your search.</div>`;
}

function showSection(id) {
  document.querySelectorAll(".section").forEach(s=>s.classList.toggle("active",s.id===id));
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.section===id));
  const titles={dashboard:["Security overview","OVERVIEW","Network intrusion detection powered by a trained NSL-KDD classifier."],detection:["Traffic Detection","DETECTION","Run the trained model against a complete 41-feature connection."],analytics:["Analytics","ANALYTICS","Evaluation results and feature-level model insights."],performance:["Model Performance","PERFORMANCE","Compare the candidate classifiers used during model selection."],history:["Detection History","HISTORY","Review predictions made during this browser session."]};
  const t=titles[id]||titles.dashboard; $("pageTitle").textContent=t[0]; $("crumbText").textContent=t[1]; $("pageSubtitle").textContent=t[2];
  if(window.innerWidth<780) $("sidebar").classList.remove("open");
}

function loadSample(kind) {
  recordFromForm();
  state.records[state.activeRecord] = (kind === "attack" ? attack : normal).slice();
  renderInputs(state.records[state.activeRecord]);
  showSection("detection"); hideError(); $("result").classList.add("hidden"); $("resultEmpty").classList.remove("hidden"); $("batchResults").classList.add("hidden");
  showToast("Sample loaded",kind === "attack" ? "Real NSL-KDD DoS/Neptune record loaded." : "Real NSL-KDD normal record loaded.");
}

function loadMixedSamples() {
  state.records = [normal.slice(), attack.slice(), normal.slice(), attack.slice(), normal.slice()];
  state.activeRecord = 0;
  renderInputs(state.records[0]);
  showSection("detection"); hideError(); $("result").classList.add("hidden"); $("resultEmpty").classList.remove("hidden"); $("batchResults").classList.add("hidden");
  showToast("5 samples loaded", "A mixed normal/attack queue is ready for batch classification.");
}

document.querySelectorAll(".nav-item").forEach(b=>b.addEventListener("click",()=>showSection(b.dataset.section)));
document.querySelectorAll("[data-go]").forEach(b=>b.addEventListener("click",()=>showSection(b.dataset.go)));
document.querySelectorAll("[data-sample]").forEach(b=>b.addEventListener("click",()=>loadSample(b.dataset.sample)));
$("predictionForm").addEventListener("submit",e=>{e.preventDefault();predict()});
$("addRecord").addEventListener("click",()=>addRecord());
$("duplicateRecord").addEventListener("click",()=>addRecord(state.records[state.activeRecord]));
$("clearRecords").addEventListener("click",()=>resetRecords());
$("predictAll").addEventListener("click",predictAll);
$("loadMixed").addEventListener("click",loadMixedSamples);
$("healthBtn").addEventListener("click",loadMetadata);
$("errorClose").addEventListener("click",hideError);
$("historySearch").addEventListener("input",renderHistory);
$("clearHistory").addEventListener("click",()=>{state.history=[];localStorage.removeItem("cypherx_history");renderHistory();showToast("History cleared","Local detection history was removed.")});
$("menuBtn").addEventListener("click",()=>$("sidebar").classList.toggle("open"));

renderInputs(normal);
loadMetadata();
setInterval(loadMetadata,15000);
