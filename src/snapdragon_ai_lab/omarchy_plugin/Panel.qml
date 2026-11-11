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
    property string view: "models"
    property bool pollingPaused: false
    property bool actionFailed: false
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
    readonly property string loadedShort: root.vm.loaded !== "" ? strings.t("model." + root.vm.loaded + ".short") : ""
    readonly property string activeShort: root.vm.loaded !== "" ? root.loadedShort
        : root.vm.selected !== "" ? strings.t("model." + root.vm.selected + ".short") : ""

    implicitWidth: button.implicitWidth
    implicitHeight: button.implicitHeight

    Commons.Translations { id: strings; directory: Qt.resolvedUrl("i18n") }

    function applyStatus(text) {
        var next = Model.parse(text, root.loadingSince)
        if (next.phase !== "loading") root.loadingSince = 0
        else if (root.loadingSince === 0) root.loadingSince = Date.now()
        root.nowMs = Date.now()
        root.remainingSeconds = next.idle.remainingSeconds
        root.vm = next
    }

    function refresh() { if (!statusProc.running) statusProc.running = true }
    function refreshLedger() { if (!ledgerProc.running) ledgerProc.running = true }
    function refreshStorage() { if (!storageProc.running) storageProc.running = true }

    // Actions run one at a time. A queue item with `prepare` set is a harness
    // warm-up: its name is in `preparing` while it runs.
    function prepareItem(name) {
        return { args: ["harness-prepare", "--harness", name], prepare: name }
    }
    function startActions(list) {
        if (list.length === 0 || root.busy || root.queue.length > 0) return
        root.actionFailed = false
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
            if (root.view === "storage") root.refreshStorage()
        }
    }
    onViewChanged: if (root.view === "storage") root.refreshStorage()

    Process {
        id: statusProc
        command: [root.controlExecutable, "status", "--json"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.applyStatus(text) }
    }

    Process {
        id: ledgerProc
        command: [root.controlExecutable, "ledger", "--json", "--limit", "50"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.ledger = Model.parseLedger(text) }
        onExited: function(code) { if (code !== 0) root.ledger = Model.parseLedger("") }
    }

    Process {
        id: storageProc
        command: [root.controlExecutable, "storage", "--json"]
        stdout: StdioCollector { waitForEnd: true; onStreamFinished: root.storage = Model.parseStorage(text) }
        onExited: function(code) { if (code !== 0) root.storage = Model.parseStorage("") }
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
            }
            root.refresh()
            if (root.view === "storage") root.refreshStorage()
            if (root.view === "activity") root.refreshLedger()
            if (root.queue.length > 0) Qt.callLater(root.pump)
        }
    }

    Timer {
        id: statusTimer
        interval: root.opened && root.vm.phase === "loading" ? 1000 : 5000
        running: root.setting("pollEnabled", true) && !root.pollingPaused
        repeat: true
        triggeredOnStart: true
        onTriggered: root.refresh()
    }

    Timer {
        id: ledgerTimer
        interval: 5000
        running: root.opened && root.view === "activity" && root.setting("pollEnabled", true) && !root.pollingPaused
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
        tooltipText: root.vm.phase === "offline" || root.vm.phase === "sleeping" ? strings.t("bar." + root.vm.phase)
            : strings.t("bar." + root.vm.phase).arg(root.activeShort)
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
                spacing: Style.space(10)

                Text {
                    textFormat: Text.PlainText
                    text: strings.t("title")
                    color: root.foreground
                    font.family: root.face
                    font.pixelSize: Style.font.title
                }

                NowCard {
                    width: column.width
                    vm: root.vm
                    strings: strings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    remainingSeconds: root.remainingSeconds
                    loadingSeconds: root.loadingSeconds
                    onLoadRequested: root.act(["load"])
                    onUnloadRequested: root.act(["unload"])
                }

                Ui.ButtonGroup {
                    options: [
                        { value: "models", label: strings.t("tab.models") },
                        { value: "activity", label: strings.t("tab.activity") },
                        { value: "storage", label: strings.t("tab.storage") }
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
                    strings: strings
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
                }

                ActivityView {
                    id: activityView
                    visible: root.view === "activity"
                    width: column.width
                    ledger: root.ledger
                    vm: root.vm
                    strings: strings
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
                    vm: root.vm
                    strings: strings
                    foreground: root.foreground
                    accent: root.accent
                    urgent: root.urgent
                    face: root.face
                    busy: root.busy
                    onDeleteRequested: function(id) { root.act(["storage-delete", "--id", id]) }
                    onClearRequested: function(prompts, conversations, harnesses) {
                        root.clearCaches(prompts, conversations, harnesses)
                    }
                }

                Ui.PanelSeparator {
                    width: column.width
                    foreground: root.foreground
                }

                Row {
                    width: column.width
                    spacing: Style.space(6)
                    Ui.Button {
                        width: (column.width - Style.space(6) * 2) / 3
                        text: strings.t("launch.chat")
                        enabled: !root.busy
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["chat"])
                    }
                    Ui.Button {
                        width: (column.width - Style.space(6) * 2) / 3
                        text: strings.t("launch.code")
                        enabled: !root.busy
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["code"])
                    }
                    Ui.Button {
                        width: (column.width - Style.space(6) * 2) / 3
                        text: strings.t("launch.studio")
                        enabled: !root.busy && root.vm.images !== "generating"
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: root.act(["images"])
                    }
                }

                Text {
                    visible: root.vm.images === "generating"
                    width: column.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: visible ? strings.t("launch.studioBusy").arg(root.vm.imageModel) : ""
                    color: root.foreground
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    visible: root.vm.loaded !== ""
                    width: column.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: strings.t("launch.studioNote")
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.caption
                }

                Row {
                    width: column.width
                    spacing: Style.space(6)
                    Text {
                        id: pollText
                        width: column.width - pauseButton.implicitWidth - Style.space(6)
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                        text: strings.t("footer.polling")
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                    Ui.Button {
                        id: pauseButton
                        text: root.pollingPaused ? strings.t("footer.resume") : strings.t("footer.pause")
                        bordered: true
                        foreground: root.foreground
                        accent: root.accent
                        fontFamily: root.face
                        onClicked: {
                            root.pollingPaused = !root.pollingPaused
                            if (!root.pollingPaused) root.refresh()
                        }
                    }
                }

                Text {
                    visible: root.actionFailed
                    width: column.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: visible ? strings.t("action.failed") : ""
                    color: root.urgent
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    visible: root.harnessFailed !== ""
                    width: column.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: visible ? strings.t("harness.failed").arg(root.harnessFailed) : ""
                    color: root.urgent
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }
            }
        }
    }
}
