import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui
import "Model.js" as Model

Ui.Panel {
    id: root
    readonly property string controlExecutable: (Quickshell.env("HOME") || "") + "/.local/bin/local-ai"
    moduleName: "snapdragon.local-ai"
    ipcTarget: "snapdragon.local-ai"

    property var vm: Model.parse("", 0)
    property var ledger: Model.parseLedger("")
    property var storage: Model.parseStorage("")
    property var cache: Model.parseCache("")
    property var share: Model.parseShare("")
    property string view: "models"
    property bool pollingPaused: false
    property bool actionFailed: false
    // Seconds since the epoch; 0 = never. They feed Model.notice(), which expires them.
    property real lastActionS: 0
    property real actionFailedS: 0
    // When the last batch of panel actions finished; 0 while one runs (or none has).
    property real actionEndS: 0
    // One status poll: its output and exit code arrive separately and are applied together.
    property string statusText: ""
    property int statusCode: 0
    property bool statusHasText: false
    property bool statusHasCode: false
    property bool statusSettled: true
    property string notice: ""
    property string harnessFailed: ""
    property var preparing: []
    property var queue: []
    property var current: null
    property real loadingSince: 0
    property real nowMs: Date.now()
    property int remainingSeconds: -1

    readonly property bool busy: actionProc.running
    readonly property int loadingSeconds: root.loadingSince > 0
        ? Math.max(0, Math.floor((root.nowMs - root.loadingSince) / 1000)) : -1
    readonly property color foreground: bar ? bar.foreground : Commons.Color.foreground
    readonly property color accent: Commons.Color.accent
    readonly property color urgent: Commons.Color.urgent
    readonly property string face: bar ? bar.fontFamily : Style.font.family
    readonly property string loadedShort: root.vm.loaded !== "" ? localAiStrings.t("model." + root.vm.loaded + ".short") : ""
    readonly property string activeShort: root.vm.loaded !== "" ? root.loadedShort
        : root.vm.selected !== "" ? localAiStrings.t("model." + root.vm.selected + ".short") : ""

    // The loaded model's last real answer, for the hero's speed figure.
    readonly property var lastAnswer: Model.lastAnswer(root.ledger, root.vm.loaded)

    implicitWidth: button.implicitWidth
    implicitHeight: button.implicitHeight

    Commons.Translations { id: localAiStrings; directory: Qt.resolvedUrl("i18n") }

    // The status command prints its answer and exits 0, or 1 when the service is not running.
    // Anything else (no answer, another exit code, a command that never started) is the calm
    // "error" state with its reason; Model.parse decides which.
    function settleStatus(force) {
        if (root.statusSettled || (!force && !(root.statusHasText && root.statusHasCode))) return
        var text = root.statusHasText ? root.statusText : ""
        var code = root.statusHasCode ? root.statusCode : -1
        root.statusSettled = true
        root.statusText = ""
        root.statusHasText = false
        root.statusHasCode = false
        root.applyStatus(text, code)
    }

    function applyStatus(text, code) {
        var next = Model.parse(text, root.loadingSince, code)
        if (next.phase !== "loading") root.loadingSince = 0
        else if (root.loadingSince === 0) root.loadingSince = Date.now()
        root.nowMs = Date.now()
        root.remainingSeconds = next.idle.remainingSeconds
        var wasAnswering = root.vm.phase === "answering"
        root.vm = next
        if (wasAnswering && next.phase !== "answering") root.refreshLedger()
        root.updateNotice()
    }

    function applyLedger(text) {
        root.ledger = Model.parseLedger(text)
        root.updateNotice()
    }

    function updateNotice() {
        root.notice = Model.notice(root.vm, root.ledger, Date.now() / 1000, root.lastActionS, root.actionFailedS,
                                   root.actionEndS)
    }

    function refresh() {
        if (statusProc.running) return
        root.statusText = ""
        root.statusHasText = false
        root.statusHasCode = false
        root.statusSettled = false
        statusProc.running = true
    }
    function refreshLedger() { if (!ledgerProc.running) ledgerProc.running = true }
    function refreshStorage() { if (!storageProc.running) storageProc.running = true }
    function refreshCache() { if (!cacheProc.running) cacheProc.running = true }
    function refreshShare() { if (!shareProc.running) shareProc.running = true }
    // Sharing with other devices: off, tailscale or lan.
    function setShare(mode) { root.act(mode === "off" ? ["share", "off"] : ["share", "on", mode]) }

    // Actions run one at a time. A queue item with `prepare` set is a harness
    // warm-up: its name is in `preparing` while it runs.
    function prepareItem(name) {
        return { args: ["harness-prepare", "--harness", name], prepare: name }
    }
    function startActions(list) {
        if (list.length === 0 || root.busy || root.queue.length > 0) return
        root.actionFailed = false
        root.actionFailedS = 0
        root.actionEndS = 0
        root.lastActionS = Date.now() / 1000
        root.updateNotice()
        root.queue = list
        root.pump()
    }
    function pump() {
        if (actionProc.running || root.queue.length === 0) return
        var next = root.queue[0]
        root.queue = root.queue.slice(1)
        root.current = next
        if (next.prepare !== "") root.preparing = root.preparing.concat([next.prepare])
        actionProc.command = [root.controlExecutable].concat(next.args)
        actionProc.running = true
    }
    function act(args) { root.startActions([{ args: args, prepare: "" }]) }
    function prepareHarness(name) {
        if (root.preparing.indexOf(name) < 0) root.startActions([root.prepareItem(name)])
    }
    function prepareAll() {
        var list = []
        var rows = root.vm.harnesses || []
        for (var i = 0; i < rows.length; i++)
            if (rows[i].state === "warmup" && root.preparing.indexOf(rows[i].name) < 0)
                list.push(root.prepareItem(rows[i].name))
        root.startActions(list)
    }
    // Conversation cache: budget in whole GB (0 = off), or clear every saved conversation.
    // Context mode of the selected 35B MoE model: fast (the default), deep or max.
    function setMode(mode) { root.act(["mode", "--mode", mode]) }
    function setCacheBudget(gb) { root.act(["cache", "budget", String(gb), "--json"]) }
    function clearConversationCache() { root.act(["cache", "clear", "--all", "--json"]) }
    function clearCaches(prompts, conversations, harnesses) {
        var list = []
        if (prompts) list.push({ args: ["clear-cache"], prepare: "" })
        if (conversations) list.push({ args: ["unload"], prepare: "" })
        if (harnesses) list.push({ args: ["harness-forget"], prepare: "" })
        root.startActions(list)
    }

    onOpenedChanged: {
        if (opened) {
            root.refresh()
            root.refreshLedger()
            root.refreshShare()
            if (root.view === "storage") { root.refreshStorage(); root.refreshCache() }
        }
    }
    onViewChanged: if (root.view === "storage") { root.refreshStorage(); root.refreshCache(); root.refreshShare() }

    Process {
        id: statusProc
        command: [root.controlExecutable, "status", "--json"]
        stdout: StdioCollector {
            waitForEnd: true
            onStreamFinished: {
                root.statusText = text
                root.statusHasText = true
                root.settleStatus(false)
            }
        }
        onExited: function(code) {
            root.statusCode = code
            root.statusHasCode = true
            root.settleStatus(false)
        }
        // A command that never started gives no output and may give no exit code either. Once
        // the process is down, a short grace lets the normal pair arrive; then whatever has
        // come is applied (nothing at all = the command is missing).
        onRunningChanged: if (!statusProc.running && !root.statusSettled) statusGrace.restart()
    }

    Timer {
        id: statusGrace
        interval: 500
        onTriggered: if (!statusProc.running) root.settleStatus(true)
    }

    Process {
        id: ledgerProc
        command: [root.controlExecutable, "ledger", "--json", "--limit", "50"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.applyLedger(text) }
        onExited: function(code) { if (code !== 0) root.applyLedger("") }
    }

    Process {
        id: storageProc
        command: [root.controlExecutable, "storage", "--json"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.storage = Model.parseStorage(text) }
        onExited: function(code) { if (code !== 0) root.storage = Model.parseStorage("") }
    }

    Process {
        id: cacheProc
        command: [root.controlExecutable, "cache", "status", "--json"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.cache = Model.parseCache(text) }
        onExited: function(code) { if (code !== 0) root.cache = Model.parseCache("") }
    }

    // An older control command has no share state: it prints nothing and the row stays hidden.
    Process {
        id: shareProc
        command: [root.controlExecutable, "share", "status", "--json"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.share = Model.parseShare(text) }
    }

    Process {
        id: actionProc
        onExited: function(code) {
            var item = root.current
            root.current = null
            if (item && item.prepare !== "") {
                var name = item.prepare
                root.preparing = root.preparing.filter(function(n) { return n !== name })
                if (code !== 0) {
                    root.harnessFailed = name
                    harnessFailedTimer.restart()
                }
            } else {
                root.actionFailed = code !== 0
                root.actionFailedS = code !== 0 ? Date.now() / 1000 : 0
            }
            if (root.queue.length === 0) root.actionEndS = Date.now() / 1000
            root.refresh()
            root.updateNotice()
            if (root.view === "storage") { root.refreshStorage(); root.refreshCache() }
            if (root.opened) { root.refreshLedger(); root.refreshShare() }
            if (root.queue.length > 0) Qt.callLater(root.pump)
        }
    }

    Timer {
        id: statusTimer
        interval: root.opened && (root.vm.phase === "loading" || root.vm.preload.state === "running") ? 1000 : 5000
        running: root.setting("pollEnabled", true) && !root.pollingPaused
        repeat: true
        triggeredOnStart: true
        onTriggered: root.refresh()
    }

    Timer {
        id: ledgerTimer
        interval: 5000
        running: root.opened && (root.view === "activity" || root.vm.phase === "answering")
            && root.setting("pollEnabled", true) && !root.pollingPaused
        repeat: true
        triggeredOnStart: true
        onTriggered: root.refreshLedger()
    }

    // Moves the loading clock and the idle countdown between polls.
    Timer {
        id: tickTimer
        interval: 1000
        running: root.opened && (root.vm.phase === "loading" || root.remainingSeconds > 0)
        repeat: true
        onTriggered: {
            root.nowMs = Date.now()
            if (root.remainingSeconds > 0) root.remainingSeconds -= 1
        }
    }

    Timer {
        id: harnessFailedTimer
        interval: 5000
        onTriggered: root.harnessFailed = ""
    }

    Ui.BarIconButton {
        id: button
        anchors.fill: parent
        bar: root.bar
        text: "\uf4ad"
        labelText: root.vm.phase === "ready" || root.vm.phase === "answering" ? root.loadedShort.charAt(0)
            : root.vm.phase === "loading" ? "…" : ""
        slotSize: Style.bar.statusSlot
        tooltipText: root.vm.phase === "offline" || root.vm.phase === "sleeping" || root.vm.phase === "error"
                ? localAiStrings.t("bar." + root.vm.phase)
            : root.vm.phase === "stopping" ? (root.vm.stopReason !== "" ? localAiStrings.t("bar.stoppingReason").arg(root.vm.stopReason)
                                                                     : localAiStrings.t("bar.stopping"))
            : localAiStrings.t("bar." + root.vm.phase).arg(root.activeShort !== "" ? root.activeShort : localAiStrings.t("bar.unlisted"))
        onPressed: root.toggle()
    }

    Ui.KeyboardPanel {
        id: panel
        anchorItem: button
        owner: root
        bar: root.bar
        open: root.opened
        focusTarget: panel.loadedContent
        contentWidth: panel.fittedContentWidth(Style.space(420))
        contentHeight: panel.fittedContentHeight(panel.loadedContentHeight)
        content: Ui.PanelKeyCatcher {
            anchors.fill: parent
            implicitHeight: column.implicitHeight
            onCloseRequested: root.close()
            onTabRequested: function(direction) { root.switchPanel(direction) }

            Column {
                id: column
                width: parent.width
                spacing: Style.space(14)

                Item {
                    width: column.width
                    height: Math.max(titleLabel.implicitHeight, pauseButton.implicitHeight)

                    Label {
                        id: titleLabel
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - pauseButton.width - pausedLabel.width - Style.space(16)
                        text: localAiStrings.t("title")
                        color: root.foreground
                        font.family: root.face
                        font.pixelSize: Style.font.title
                        font.bold: true
                    }
                    Label {
                        id: pausedLabel
                        visible: root.pollingPaused
                        width: visible ? implicitWidth : 0
                        anchors.right: pauseButton.left
                        anchors.rightMargin: Style.space(6)
                        anchors.verticalCenter: parent.verticalCenter
                        text: localAiStrings.t("footer.paused")
                        dim: true
                        color: root.foreground
                        font.family: root.face
                        font.pixelSize: Style.font.bodySmall
                    }
                    Ui.Button {
                        id: pauseButton
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        iconText: root.pollingPaused ? "\uf04b" : "\uf04c"
                        iconSize: Style.font.caption
                        opacity: root.pollingPaused || hot ? 1 : 0.5
                        tooltipText: root.pollingPaused ? localAiStrings.t("footer.resume") : localAiStrings.t("footer.pause")
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: {
                            root.pollingPaused = !root.pollingPaused
                            if (!root.pollingPaused) root.refresh()
                        }
                    }
                }

                NowCard {
                    width: column.width
                    vm: root.vm
                    strings: localAiStrings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    remainingSeconds: root.remainingSeconds
                    loadingSeconds: root.loadingSeconds
                    speedTokS: root.lastAnswer !== null ? root.lastAnswer.tokS : -1
                    notice: root.harnessFailed !== "" ? "harness" : root.notice
                    noticeArg: root.harnessFailed
                    onLoadRequested: root.act(["load"])
                    onUnloadRequested: root.act(["unload"])
                    onDetailsRequested: root.view = "activity"
                }

                Ui.ButtonGroup {
                    options: [
                        { value: "models", label: localAiStrings.t("tab.models") },
                        { value: "activity", label: localAiStrings.t("tab.activity") },
                        { value: "storage", label: localAiStrings.t("tab.storage") }
                    ]
                    value: root.view
                    foreground: root.foreground
                    accent: root.accent
                    fontFamily: root.face
                    onChanged: function(picked) { root.view = picked }
                }

                ModelsView {
                    id: modelsView
                    visible: root.view === "models"
                    width: column.width
                    vm: root.vm
                    strings: localAiStrings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    preparing: root.preparing
                    onSelectRequested: function(key) { root.act([key]) }
                    onPrepareRequested: function(name) { root.prepareHarness(name) }
                    onPrepareAllRequested: root.prepareAll()
                    onForgetRequested: function(name) { root.act(["harness-forget", "--harness", name]) }
                    onFlashCacheRequested: function(preference) { root.act(["flash-cache", "--pref", preference]) }
                    onModeRequested: function(mode) { root.setMode(mode) }
                }

                ActivityView {
                    id: activityView
                    visible: root.view === "activity"
                    width: column.width
                    ledger: root.ledger
                    vm: root.vm
                    strings: localAiStrings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                }

                StorageView {
                    id: storageView
                    visible: root.view === "storage"
                    width: column.width
                    storage: root.storage
                    cache: root.cache
                    vm: root.vm
                    strings: localAiStrings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    onBudgetRequested: function(gb) { root.setCacheBudget(gb) }
                    onCacheClearRequested: root.clearConversationCache()
                    onDeleteRequested: function(id) { root.act(["storage-delete", "--id", id]) }
                    onClearRequested: function(prompts, conversations, harnesses) {
                        root.clearCaches(prompts, conversations, harnesses)
                    }
                }

                ShareRow {
                    visible: root.view === "storage" && root.share.available === true
                    width: column.width
                    share: root.share
                    strings: localAiStrings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    onShareRequested: function(mode) { root.setShare(mode) }
                }

                Ui.PanelSeparator {
                    width: column.width
                    foreground: root.foreground
                }

                Row {
                    id: launchRow
                    width: column.width
                    spacing: Style.space(8)

                    readonly property real cell: (width - spacing * 2) / 3

                    Ui.Button {
                        width: launchRow.cell
                        iconText: "\uf086"
                        text: localAiStrings.t("launch.chat")
                        enabled: !root.busy
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["chat"])
                    }
                    Ui.Button {
                        width: launchRow.cell
                        iconText: "\uf121"
                        text: localAiStrings.t("launch.code")
                        enabled: !root.busy
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["code"])
                    }
                    Ui.Button {
                        width: launchRow.cell
                        iconText: "\uf03e"
                        text: localAiStrings.t("launch.studio")
                        tooltipText: root.vm.images === "generating" ? localAiStrings.t("launch.studioBusy").arg(root.vm.imageModel)
                            : root.vm.loaded !== "" ? localAiStrings.t("launch.studioNote") : ""
                        enabled: !root.busy && root.vm.images !== "generating"
                        opacity: root.vm.images === "generating" ? 0.5 : 1
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["images"])
                    }
                }
            }
        }
    }
}
