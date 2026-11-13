import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui
import "Model.js" as Model

// The hero: what is loaded, its last speed, RAM, and the one load/unload action.
Rectangle {
    id: root

    property var vm: Model.parse("", 0)
    property var strings: null
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property color urgent: Commons.Color.urgent
    property string face: Style.font.family
    property bool busy: false
    property int remainingSeconds: -1
    property int loadingSeconds: -1
    // tok/s of the loaded model's last answer, -1 when unknown.
    property real speedTokS: -1
    // "" or one of failure / action / start / request / harness; noticeArg fills the harness name.
    property string notice: ""
    property string noticeArg: ""

    signal loadRequested()
    signal unloadRequested()
    signal detailsRequested()

    component T: Label { color: root.foreground; font.family: root.face }

    function fmt(value, decimals) { return Commons.I18n.formatNumber(value, decimals) }
    function gibText(value) { return root.strings.t("unit.gib").arg(root.fmt(value, 1)) }

    function rowFor(key) {
        var rows = root.vm.models || []
        for (var i = 0; i < rows.length; i++)
            if (rows[i].key === key) return rows[i]
        return null
    }

    function nameOf(key) { return key ? root.strings.t("model." + key + ".short") : "" }

    readonly property string phase: root.vm.phase
    readonly property string loadedName: root.nameOf(root.vm.loaded)
    readonly property string selectedName: root.nameOf(root.vm.selected)
    // A headline is never blank: a model the panel has no name for is still "a model".
    readonly property string shownName: root.loadedName !== "" ? root.loadedName
        : root.phase === "loading" && root.selectedName !== "" ? root.selectedName : root.strings.t("now.unlisted")
    readonly property bool up: root.phase === "ready" || root.phase === "answering"
    // "running" / "done" / "": the last session being restored into the loaded model.
    readonly property string preload: Model.preloadPhase(root.vm)

    readonly property bool showLoad: root.vm.canLoad === true && root.vm.selected !== ""
    readonly property bool showUnload: root.up
    readonly property bool actionEnabled: root.showLoad ? !root.busy : (root.vm.canUnload && !root.busy)
    readonly property bool hasActionButton: root.showLoad || root.showUnload

    // The selected model's row while it is not the loaded one; drives the "needs" stat and ghost bar.
    readonly property var pending: {
        var row = root.rowFor(root.vm.selected)
        return row && root.vm.selected !== root.vm.loaded && (row.needGib || -1) > 0 ? row : null
    }

    readonly property string headline: {
        switch (root.phase) {
        case "error": return root.strings.t("now.error")
        case "offline": return root.strings.t("now.offline")
        case "stopping": return root.strings.t("now.stopping")
        case "loading":
        case "ready":
        case "answering": return root.shownName
        default: return root.strings.t("now.sleeping")
        }
    }

    readonly property string detail: {
        switch (root.phase) {
        case "error":
            return root.vm.fault === "command" ? root.strings.t("now.error.command").arg(root.fmt(root.vm.faultCode))
                : root.strings.t(root.vm.fault === "missing" ? "now.error.missing" : "now.error.unreadable")
        case "offline": return root.strings.t("now.offlineDetail")
        case "sleeping":
            return root.selectedName !== "" ? root.strings.t("now.sleepingDetail").arg(root.selectedName)
                                            : root.strings.t("now.sleepingNone")
        case "loading":
            return root.loadingSeconds >= 0
                ? root.strings.t("now.loadingElapsed").arg(root.fmt(root.loadingSeconds))
                : root.strings.t("now.loading")
        case "stopping": return root.strings.t("now.stoppingDetail")
        case "ready":
            if (root.preload === "running") return root.strings.t("now.resuming")
            return root.remainingSeconds >= 0
                ? root.strings.t(root.preload === "done" ? "now.sessionCountdown" : "now.countdown")
                    .arg(Model.countdown(root.remainingSeconds))
                : root.strings.t(root.preload === "done" ? "now.sessionCountdownIdle" : "now.countdownIdle")
                    .arg(root.fmt(Math.round(root.vm.idle.totalSeconds / 60)))
        case "answering":
            return root.vm.queued > 0
                ? root.strings.tn("now.queued", root.vm.queued).arg(root.fmt(root.vm.queued))
                : root.strings.t("now.answering")
        default: return ""
        }
    }

    // What the detail line's tooltip adds; "" = no tooltip.
    readonly property string detailTip: {
        if (root.phase === "stopping")
            return root.vm.stopReason !== "" ? root.strings.t("now.stoppingReason").arg(root.vm.stopReason) : ""
        if (root.phase === "error" || root.phase === "offline") return root.strings.t("now.retryTip")
        if (root.preload === "running") return root.strings.t("now.resumingTip")
        if (root.preload === "done") {
            var p = root.vm.preload
            return p.ms >= 0 && p.src !== ""
                ? root.strings.t("now.sessionTip." + p.src).arg(root.fmt(p.ms / 1000, 1))
                : root.strings.t("now.sessionTip")
        }
        return ""
    }

    readonly property string actionText: root.showLoad ? root.strings.t("now.load").arg(root.selectedName) : root.strings.t("now.unload")
    readonly property string actionReason: root.showUnload && !root.vm.canUnload && root.vm.unloadBlocked !== ""
        ? root.strings.t("now.blocked." + root.vm.unloadBlocked) : ""

    // Up to three figures; each is a value with a caption under it.
    readonly property var stats: {
        var out = []
        var ram = root.vm.ram
        if (root.phase === "offline") return out
        if (root.up && root.speedTokS > 0)
            out.push({ value: root.strings.t("unit.toks").arg(root.fmt(root.speedTokS, root.speedTokS < 10 ? 1 : 0)),
                       label: root.strings.t("stat.speed"), bad: false })
        if (ram.modelGib >= 0)
            out.push({ value: root.gibText(ram.modelGib), label: root.strings.t("stat.model"), bad: false })
        if (ram.availableGib >= 0)
            out.push({ value: root.gibText(ram.availableGib),
                       label: ram.totalGib > 0 ? root.strings.t("stat.freeOf").arg(root.fmt(ram.totalGib, 1))
                                               : root.strings.t("stat.free"), bad: false })
        if (root.pending !== null && out.length < 3)
            out.push({ value: root.gibText(root.pending.needGib),
                       label: root.strings.t("stat.need").arg(root.selectedName), bad: root.pending.fits === "no" })
        return out
    }

    readonly property string noticeText: {
        var subject = root.selectedName !== "" ? root.selectedName : root.strings.t("now.unlisted")
        switch (root.notice) {
        case "failure":
            return root.vm.failure && root.vm.failure.reason !== "" ? root.vm.failure.reason
                : root.strings.t("notice.start").arg(root.vm.failure && root.vm.failure.model
                    ? root.nameOf(root.vm.failure.model) : subject)
        case "action": return root.strings.t("notice.action")
        case "start": return root.strings.t("notice.start").arg(subject)
        case "request": return root.strings.t("notice.request")
        case "harness": return root.strings.t("notice.harness").arg(root.noticeArg)
        default: return ""
        }
    }

    readonly property real pad: Style.space(14)

    color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.05)
    radius: Style.cornerRadius
    height: body.height + root.pad * 2

    Column {
        id: body
        x: root.pad
        y: root.pad
        width: root.width - root.pad * 2
        spacing: Style.space(12)

        Item {
            width: parent.width
            height: Math.max(headText.height, actionButton.visible ? actionButton.height : 0)

            StateDot {
                id: dot
                y: Math.round(headLine.height / 2 - height / 2)
                kind: root.phase === "answering" || root.phase === "loading" || root.preload === "running" ? "pulse"
                    : root.phase === "ready" ? "filled"
                    : "hollow"
                tint: root.up || root.phase === "loading" ? root.accent : root.foreground
            }

            Column {
                id: headText
                x: dot.width + Style.space(10)
                width: parent.width - x - (actionButton.visible ? actionButton.width + Style.space(10) : 0)
                spacing: Style.space(3)

                T {
                    id: headLine
                    width: parent.width
                    text: root.headline
                    font.pixelSize: Style.font.heading
                    font.bold: true
                }
                T {
                    width: parent.width
                    visible: root.detail !== ""
                    text: root.detail
                    dim: true
                    wrapMode: Text.WordWrap
                    maximumLineCount: 2
                    font.pixelSize: Style.font.bodySmall
                }
            }

            // The reason behind the state line (why it is stopping, how the session came back)
            // is a tooltip over the headline and its detail.
            MouseArea {
                id: stateHover
                x: headText.x
                width: headText.width
                height: headText.height
                hoverEnabled: true
                enabled: root.detailTip !== ""
            }
            Ui.PanelToolTip {
                visible: stateHover.enabled && stateHover.containsMouse
                text: root.detailTip
                width: Math.min(implicitWidth, Style.space(340))
                fontFamily: root.face
            }

            Ui.Button {
                id: actionButton
                visible: root.hasActionButton
                anchors.right: parent.right
                y: 0
                text: root.actionText
                tooltipText: root.actionReason
                opacity: root.actionEnabled ? 1 : 0.5
                bordered: true
                foreground: root.showLoad ? root.accent : root.foreground
                accent: root.accent
                fontFamily: root.face
                onClicked: {
                    if (!root.actionEnabled) return
                    if (root.showLoad) root.loadRequested()
                    else root.unloadRequested()
                }
            }
        }

        Row {
            id: statRow
            visible: root.stats.length > 0
            width: parent.width
            spacing: Style.space(12)

            Repeater {
                model: root.stats
                delegate: Column {
                    required property var modelData
                    width: (statRow.width - statRow.spacing * 2) / 3
                    spacing: Style.space(2)

                    T {
                        width: parent.width
                        text: modelData.value
                        color: modelData.bad ? root.urgent : root.foreground
                        font.pixelSize: Style.font.title
                    }
                    T {
                        width: parent.width
                        text: modelData.label
                        dim: true
                        font.pixelSize: Style.font.caption
                    }
                }
            }
        }

        RamBar {
            width: parent.width
            totalGib: root.vm.ram.totalGib
            modelGib: root.vm.ram.modelGib
            otherGib: root.vm.ram.otherGib
            needGib: root.pending !== null ? root.pending.needGib : -1
            needFits: root.pending === null || root.pending.fits !== "no"
            foreground: root.foreground
            accent: root.accent
            urgent: root.urgent
        }

        Item {
            visible: root.noticeText !== ""
            width: parent.width
            height: Math.max(noticeLabel.height, detailsButton.visible ? detailsButton.height : 0)

            T {
                id: noticeLabel
                anchors.verticalCenter: parent.verticalCenter
                width: parent.width - (detailsButton.visible ? detailsButton.width + Style.space(8) : 0)
                text: root.noticeText
                color: root.urgent
                wrapMode: Text.WordWrap
                maximumLineCount: 3
                font.pixelSize: Style.font.bodySmall
            }
            Ui.Button {
                id: detailsButton
                visible: root.notice === "request"
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                text: root.strings.t("notice.details")
                foreground: root.foreground
                accent: root.accent
                fontFamily: root.face
                fontSize: Style.font.bodySmall
                onClicked: root.detailsRequested()
            }
        }

        T {
            visible: root.vm.computeBusy && root.phase === "sleeping" && root.vm.images === "idle"
            width: parent.width
            text: root.strings.t("now.computeBusy")
            dim: true
            wrapMode: Text.WordWrap
            font.pixelSize: Style.font.bodySmall
        }
    }
}
