.pragma library

var MODEL_KEYS = ["ornith", "moe", "flash", "dense"];
var HARNESS_NAME = /^[A-Za-z0-9._-]{1,32}$/;
var STORAGE_KINDS = ["rust-weights", "rust-prefix", "flash-sidecar", "llama-prefix", "llama-repack"];
var STORAGE_STATES = ["in-use", "referenced", "protected", "unreferenced"];

function oneOf(list, value, fallback) {
    return list.indexOf(value) >= 0 ? value : fallback;
}

function keyOf(value) {
    return oneOf(MODEL_KEYS, value, "");
}

// A finite number in 0..max, else -1 (unknown).
function num(n, max) {
    return typeof n === "number" && isFinite(n) && n >= 0 && n <= max ? n : -1;
}

// A whole number in 0..max, else 0.
function count(n, max) {
    return typeof n === "number" && isFinite(n) && Math.floor(n) === n && n >= 0 && n <= max ? n : 0;
}

function parse(text, loadingSince) {
    if (typeof text !== "string" || text.length > 16384) return build({});
    var value;
    try {
        value = JSON.parse(text);
    } catch (_) {
        return build({});
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) return build({});
    return build(value);
}

function statusRow(rows, key) {
    for (var i = 0; i < rows.length; i++) {
        var row = rows[i];
        if (row && typeof row === "object" && row.key === key) return row;
    }
    return null;
}

function flashCache(value, canChange) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return null;
    var pref = oneOf(["auto", "20", "18", "16", "14"], value.pref, "");
    if (pref === "") return null;
    var tier = oneOf([20, 18, 16, 14], value.tier_gib, -1);
    var reason = oneOf(["auto", "preferred", "preferred-does-not-fit", "preferred-swap-tolerant", "refuse"], value.reason, "");
    return {pref: pref, tierGib: tier, reason: reason,
            control: value.preference_control === true,
            canChange: value.preference_control === true && canChange,
            workingSetGib: num(value.estimated_working_set_gib, 128)};
}

function build(value) {
    var service = oneOf(["unloaded", "ready", "loading", "offline"], value.chat_state, "offline");
    var active = count(value.active_requests, 4096);
    var queued = count(value.queued_requests, 128);
    var images = oneOf(["idle", "generating", "offline"], value.image_state, "offline");
    var selected = keyOf(value.selected_model);
    var loaded = keyOf(value.loaded_model);
    if (loaded === "" && (service === "ready" || service === "loading")) loaded = keyOf(value.model);

    var phase = service === "offline" ? "offline" : service === "unloaded" ? "sleeping" : service === "loading" ? "loading" : active > 0 ? "answering" : "ready";
    var unloadBlocked = service === "unloaded" || service === "offline" ? "nothing" : service === "loading" ? "loading" : active > 0 ? "answering" : images === "generating" ? "images" : "";

    var totalGib = num(value.total_gib, 4096) > 0 ? value.total_gib : -1;
    var availableGib = num(value.available_gib, 1024);
    var modelGib = num(value.resident_gib, 4096) > 0 ? value.resident_gib : -1;
    var otherGib = totalGib >= 0 && availableGib >= 0 ? Math.max(0, totalGib - availableGib - Math.max(0, modelGib)) : -1;
    var ttl = value.idle_seconds;
    var totalSeconds = typeof ttl === "number" && isFinite(ttl) && ttl >= 1 && ttl <= 86400 ? ttl : 600;

    var ornithAvailable = value.ornith_available === true;
    var flashAvailable = value.flash_available === true;
    var statusRows = Array.isArray(value.models) && value.models.length <= 16 ? value.models : [];

    function modelRow(key) {
        var src = statusRow(statusRows, key) || {};
        var state = "cold";
        if (key === loaded && service === "ready") state = "loaded";
        else if (service === "loading" && (key === loaded || (loaded === "" && key === selected))) state = "loading";
        var workingSet = src.memory_basis === 'qualified_working_set' ? num(src.memory_gib, 128) : -1;
        var needGib = workingSet < 0 && num(src.memory_gib, 1024) > 0 ? src.memory_gib : -1;
        var fits = "unknown";
        if (needGib >= 0 && availableGib >= 0) {
            fits = needGib <= availableGib + Math.max(0, modelGib) || key === loaded ? "yes" : "no";
        }
        var result = {
            key: key,
            group: key === "flash" ? "heavy" : "named",
            available: key === "ornith" ? ornithAvailable : key === "flash" ? flashAvailable : true,
            selected: key === selected,
            loaded: state === "loaded",
            isDefault: key === "ornith",
            state: state,
            engine: oneOf(["rust", "llama", "stream"], src.engine, ""),
            prefix: oneOf(["warm", "cold", "off"], src.prefix, ""),
            needGib: needGib,
            fits: fits,
            batteryStart: num(src.battery_start_percent, 100) > 0 ? src.battery_start_percent : -1
        };
        if (workingSet >= 0) result.workingSetGib = workingSet;
        var cache = key === "flash" ? flashCache(src.flash_cache,
            service !== "offline" && service !== "loading" && active === 0 && queued === 0) : null;
        if (cache !== null) result.cache = cache;
        return result;
    }

    var keys = ["ornith", "moe", "flash"];
    if (selected === "dense" || loaded === "dense") keys.push("dense");
    var models = [];
    for (var k = 0; k < keys.length; k++) models.push(modelRow(keys[k]));

    var harnesses = [];
    var hasHarnessData = Array.isArray(value.harnesses) && value.harnesses.length <= 256;
    if (hasHarnessData) {
        for (var i = 0; i < value.harnesses.length && harnesses.length < 12; i++) {
            var h = value.harnesses[i];
            if (!h || typeof h !== "object" || typeof h.name !== "string" || !HARNESS_NAME.test(h.name)) continue;
            var cached = count(h.cache_n, 1000000000);
            var evaluated = count(h.prompt_n, 1000000000);
            harnesses.push({
                name: h.name,
                state: h.warm === true ? "ready" : "warmup",
                estimateSeconds: harnessEstimate(cached + evaluated),
                tools: count(h.tools, 1024),
                lastUsed: count(h.last_used, 4102444800),
                cachedTokens: cached,
                evaluatedTokens: evaluated,
                prefixes: count(h.prefixes, 1024)
            });
        }
    }

    var flashTotal = count(value.flash_download_total_bytes, 2199023255552);
    var flashDone = count(value.flash_download_bytes, flashTotal);
    var flashShards = count(value.flash_download_shards, 16);
    var flashVerified = count(value.flash_verified_shards, flashShards);
    var flashActive = value.flash_download_active === true;
    var flashState = oneOf(["absent", "downloading", "verifying", "weights-ready-testing", "paused"], value.flash_download_state, "absent");
    var trialElapsed = value.flash_trial_elapsed_seconds;
    var trialValid = typeof trialElapsed === "number" && isFinite(trialElapsed) && Math.floor(trialElapsed) === trialElapsed && trialElapsed >= 0 && trialElapsed <= 86400;
    var trialKind = oneOf(["coding", "http", "prefill", "memory", "overlap", "baseline"], value.flash_trial_kind, "");
    var trialActive = value.flash_trial_active === true && trialKind !== "" && trialValid;
    if (flashTotal === 0 || flashShards === 0) flashState = "absent";
    else if ((flashState === "downloading" || flashState === "verifying") && !flashActive) flashState = "paused";
    else if (flashState === "weights-ready-testing" && (flashVerified !== flashShards || flashDone !== flashTotal)) flashState = "paused";

    return {
        service: service,
        phase: phase,
        active: active,
        queued: queued,
        selected: selected,
        loaded: loaded,
        backend: oneOf(["cpu", "npu", "gpu"], value.backend, "cpu"),
        ram: {totalGib: totalGib, availableGib: availableGib, modelGib: modelGib, otherGib: otherGib},
        idle: {
            totalSeconds: totalSeconds,
            remainingSeconds: num(value.idle_remaining_seconds, 86400)
        },
        canUnload: unloadBlocked === "",
        unloadBlocked: unloadBlocked,
        models: models,
        harnesses: harnesses,
        hasHarnessData: hasHarnessData,
        images: images,
        imageModel: value.image_model === "q4" ? "Q4" : value.image_model === "q8" ? "Q8" : "",
        computeBusy: value.compute_busy === true,
        hasError: value.has_error === true,
        prefixCache: value.prefix_cache_enabled === true,
        flash: {
            state: flashState,
            percent: flashTotal > 0 ? Math.floor(1000 * flashDone / flashTotal) / 10 : 0,
            doneBytes: flashDone,
            totalBytes: flashTotal,
            verified: flashVerified,
            shards: flashShards,
            waiting: flashState === "verifying" && value.flash_waiting_verification === true,
            trialActive: trialActive,
            trialKind: trialActive ? trialKind : "",
            trialElapsed: trialActive ? trialElapsed : 0
        }
    };
}

function median(values) {
    if (values.length === 0) return -1;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function parseLedger(text) {
    var fallback = {available: false, rows: [], summary: {count: 0, failed: 0, medianTtftMs: -1, medianTokS: -1, cacheShare: -1, lastT: 0}};
    if (typeof text !== "string" || text.length > 262144) return fallback;
    var value;
    try {
        value = JSON.parse(text);
    } catch (_) {
        return fallback;
    }
    if (!value || typeof value !== "object" || value.ledger_available !== true || !Array.isArray(value.rows)) return fallback;

    var rows = [], ttfts = [], speeds = [], failed = 0, lastT = 0, cacheSum = 0, promptSum = 0;
    var source = value.rows.slice(0, 200);
    for (var i = 0; i < source.length; i++) {
        var row = source[i];
        if (!row || typeof row !== "object") continue;
        var status = num(row.status, 999);
        var queue = num(row.queue_wait_ms, 1e15);
        var admit = num(row.admit_ms, 1e15);
        var ttfb = num(row.ttfb_ms, 1e15);
        var wall = num(row.wall_ms, 1e15);
        var predictedN = num(row.predicted_n, 1e15);
        var predictedMs = num(row.predicted_ms, 1e15);
        var cacheN = num(row.cache_n, 1e15);
        var promptN = num(row.prompt_n, 1e15);
        var t = num(row.t, 1e15);
        var ttft = ttfb < 0 ? -1 : (queue < 0 ? 0 : queue) + (admit < 0 ? 0 : admit) + ttfb;
        var tokS = predictedN > 0 && predictedMs > 0 ? predictedN / predictedMs * 1000 : -1;
        var known = cacheN >= 0 && promptN >= 0;
        var cacheShare = known && cacheN + promptN > 0 ? cacheN / (cacheN + promptN) : -1;
        var ok = status >= 200 && status <= 299;
        if (status >= 0 && !ok) failed++;
        if (known) {
            cacheSum += cacheN;
            promptSum += promptN;
        }
        if (ttft >= 0) ttfts.push(ttft);
        if (tokS >= 0) speeds.push(tokS);
        if (t > lastT) lastT = t;
        rows.push({
            t: t,
            model: keyOf(row.model),
            harness: typeof row.harness === "string" && HARNESS_NAME.test(row.harness) ? row.harness : "",
            ok: ok,
            status: status,
            queueMs: queue,
            loadMs: admit >= 1000 ? admit : 0,
            ttftMs: ttft,
            wallMs: wall,
            tokS: tokS,
            cacheN: cacheN,
            promptN: promptN,
            cacheShare: cacheShare,
            predictedN: predictedN,
            finish: oneOf(["stop", "length", "tool_calls"], row.finish, "")
        });
    }
    return {
        available: true,
        rows: rows,
        summary: {
            count: rows.length,
            failed: failed,
            medianTtftMs: median(ttfts),
            medianTokS: median(speeds),
            cacheShare: cacheSum + promptSum > 0 ? cacheSum / (cacheSum + promptSum) : -1,
            lastT: lastT
        }
    };
}

function parseStorage(text) {
    var fallback = {available: false, entries: [], totalBytes: 0, freeableBytes: 0};
    if (typeof text !== "string" || text.length > 65536) return fallback;
    try {
        var value = JSON.parse(text);
        if (!value || typeof value !== "object" || value.storage_available !== true || !Array.isArray(value.entries) || value.entries.length > 128) return fallback;
        var entries = [], total = 0, freeable = 0;
        for (var i = 0; i < value.entries.length; i++) {
            var row = value.entries[i];
            if (!row || typeof row !== "object" || STORAGE_KINDS.indexOf(row.kind) < 0 || STORAGE_STATES.indexOf(row.state) < 0) continue;
            if (typeof row.id !== "string" || !/^[a-z]+(-[a-z]+)*-[0-9a-f]{16}$/.test(row.id)) continue;
            var size = typeof row.bytes === "number" && isFinite(row.bytes) && row.bytes >= 0 && Math.floor(row.bytes) === row.bytes ? row.bytes : 0;
            var deletable = row.deletable === true && row.state === "unreferenced";
            entries.push({
                id: row.id, kind: row.kind, state: row.state, bytes: size, deletable: deletable,
                name: typeof row.name === "string" && /^[A-Za-z0-9._-]{1,96}$/.test(row.name) ? row.name : "",
                model: MODEL_KEYS.indexOf(row.model) >= 0 || row.model === "other" ? row.model : ""
            });
            total += size;
            if (deletable) freeable += size;
        }
        return {available: true, entries: entries, totalBytes: total, freeableBytes: freeable};
    } catch (_) { return fallback; }
}

// Whole seconds as m:ss, minutes unbounded; "" for anything that is not a number >= 0.
function countdown(seconds) {
    if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) return "";
    var whole = Math.floor(seconds);
    var minutes = Math.floor(whole / 60);
    var rest = whole % 60;
    return minutes + ":" + (rest < 10 ? "0" : "") + rest;
}

function gib(bytes) {
    if (typeof bytes !== "number" || !isFinite(bytes) || bytes < 0) return 0;
    return Math.round(bytes / 1073741824 * 10) / 10;
}

function harnessEstimate(tokens) {
    if (typeof tokens !== "number" || !isFinite(tokens) || tokens <= 0) return -1;
    return Math.ceil(tokens / 124 / 5) * 5;
}
