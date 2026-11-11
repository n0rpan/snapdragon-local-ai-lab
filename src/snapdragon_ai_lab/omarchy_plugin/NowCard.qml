import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui
import "Model.js" as Model

// What is loaded now: state, countdown, RAM, and the one load/unload action.
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

    signal loadRequested()
    signal unloadRequested()

    // A dot that is filled, hollow (ring) or pulsing; the words carry the state too.
    component StateMark: Item {
        id: mark
        property string kind: "filled"
        property color tint: Commons.Color.foreground
        width: Style.space(8)
        height: width

        Rectangle {
            anchors.fill: parent
            radius: width / 2
            color: mark.kind === "hollow" ? "transparent" : mark.tint
            border.width: mark.kind === "hollow" ? 1 : 0
            border.color: mark.tint

            SequentialAnimation on opacity {
                running: mark.kind === "pulse" && !Style.reduceMotion
                loops: Animation.Infinite
                NumberAnimation { from: 1; to: 0.3; duration: Style.duration(900); easing.type: Easing.InOutQuad }
                NumberAnimation { from: 0.3; to: 1; duration: Style.duration(900); easing.type: Easing.InOutQuad }
            }
        }
    }

    function fmt(value, decimals) {
        return Commons.I18n.formatNumber(value, decimals)
    }

    function rowFor(key) {
        var rows = root.vm.models || []
        for (var i = 0; i < rows.length; i++)
            if (rows[i].key === key) return rows[i]
        return null
    }

    readonly property string phase: root.vm.phase
    readonly property string loadedName: root.vm.loaded !== "" ? root.strings.t("model." + root.vm.loaded + ".short") : ""
    readonly property string selectedName: root.vm.selected !== "" ? root.strings.t("model." + root.vm.selected + ".short") : ""
    readonly property string loadingName: root.vm.loaded !== "" ? root.loadedName : root.selectedName

    readonly property bool showLoad: root.phase === "sleeping" && root.vm.selected !== ""
    readonly property bool showUnload: root.phase === "ready" || root.phase === "answering"
    readonly property bool actionEnabled: root.showLoad ? !root.busy : (root.vm.canUnload && !root.busy)
    readonly property bool hasActionButton: root.showLoad || root.showUnload

    // Selected model's need, shown as a marker on the RAM bar only while it is not loaded.
    readonly property real markerGib: {
        var row = root.rowFor(root.vm.selected)
        return row && root.vm.selected !== root.vm.loaded ? (row.needGib || -1) : -1
    }

    readonly property string headline: {
        switch (root.phase) {
        case "offline": return root.strings.t("now.offline")
        case "loading": return root.strings.t("now.loading").arg(root.loadingName)
        case "ready": return root.strings.t("now.ready").arg(root.loadedName)
        case "answering": return root.strings.t("now.answering").arg(root.loadedName)
        default: return root.strings.t("now.sleeping")
        }
    }

    readonly property string detail: {
        switch (root.phase) {
        case "offline": return root.strings.t("now.offlineDetail")
        case "sleeping": return root.strings.t("now.sleepingDetail")
        case "loading": return root.loadingSeconds >= 0 ? root.strings.t("now.loadingElapsed").arg(root.fmt(root.loadingSeconds)) : ""
        case "ready":
            return root.remainingSeconds >= 0
                ? root.strings.t("now.countdown").arg(Model.countdown(root.remainingSeconds))
                : root.strings.t("now.countdownIdle").arg(root.fmt(Math.round(root.vm.idle.totalSeconds / 60)))
        case "answering":
            return root.vm.queued > 0
                ? root.strings.tn("now.queued", root.vm.queued).arg(root.fmt(root.vm.queued))
                : root.strings.t("now.blocked.answering")
        default: return ""
        }
    }

    readonly property string actionText: root.showLoad ? root.strings.t("now.load").arg(root.selectedName) : root.strings.t("now.unload")
    readonly property string actionReason: root.showUnload && !root.vm.canUnload && root.vm.unloadBlocked !== ""
        ? root.strings.t("now.blocked." + root.vm.unloadBlocked) : ""

    readonly property bool hasModelRam: root.vm.ram.modelGib >= 0
    readonly property string freeText: root.vm.ram.totalGib > 0 && root.vm.ram.availableGib >= 0
        ? root.strings.t("now.ramFree").arg(root.fmt(root.vm.ram.availableGib, 1)).arg(root.fmt(root.vm.ram.totalGib, 1))
        : root.vm.ram.availableGib >= 0
            ? root.strings.t("now.ramFreeOnly").arg(root.fmt(root.vm.ram.availableGib, 1))
            : root.strings.t("now.ramUnknown")

    color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.06)
    border.width: 1
    border.color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.12)
    radius: Style.cornerRadius
    height: body.height + Style.space(12) * 2

    Column {
        id: body
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.margins: Style.space(12)
        spacing: Style.space(8)

        Item {
            width: parent.width
            height: Math.max(headText.implicitHeight, actionButton.visible ? actionButton.implicitHeight : 0)

            StateMark {
                id: dot
                x: 0
                y: headText.implicitHeight / 2 - height / 2
                kind: root.phase === "ready" ? "filled"
                    : root.phase === "answering" ? "pulse"
                    : root.phase === "loading" ? "pulse"
                    : "hollow"
                tint: root.phase === "ready" || root.phase === "answering" ? root.accent : root.foreground
            }

            Column {
                id: headText
                x: dot.width + Style.space(8)
                width: parent.width - x - (actionButton.visible ? actionButton.width + Style.space(8) : 0)
                spacing: Style.space(2)

                Text {
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: root.headline
                    color: root.foreground
                    font.family: root.face
                    font.pixelSize: Style.font.heading
                }
                Text {
                    width: parent.width
                    visible: root.detail !== ""
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: root.detail
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }
            }

            Ui.Button {
                id: actionButton
                visible: root.hasActionButton
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                text: root.actionText
                tooltipText: root.actionReason
                opacity: root.actionEnabled ? 1 : 0.5
                bordered: true
                foreground: root.foreground
                accent: root.accent
                fontFamily: root.face
                fontSize: Style.font.bodySmall
                onClicked: {
                    if (!root.actionEnabled) return
                    if (root.showLoad) root.loadRequested()
                    else root.unloadRequested()
                }
            }
        }

        RamBar {
            width: parent.width
            totalGib: root.vm.ram.totalGib
            modelGib: root.vm.ram.modelGib
            otherGib: root.vm.ram.otherGib
            needGib: root.markerGib
            foreground: root.foreground
            accent: root.accent
        }

        Row {
            visible: root.vm.phase !== "offline"
            Text {
                visible: root.hasModelRam
                textFormat: Text.PlainText
                text: root.strings.t("now.ramModel").arg(root.fmt(root.vm.ram.modelGib, 1))
                color: root.foreground
                font.family: root.face
                font.pixelSize: Style.font.bodySmall
            }
            Text {
                textFormat: Text.PlainText
                visible: root.hasModelRam
                text: " · "
                color: root.foreground
                font.family: root.face
                font.pixelSize: Style.font.bodySmall
            }
            Text {
                textFormat: Text.PlainText
                text: root.freeText
                color: root.foreground
                font.family: root.face
                font.pixelSize: Style.font.bodySmall
            }
        }

        Text {
            visible: root.vm.hasError
            width: parent.width
            wrapMode: Text.WordWrap
            textFormat: Text.PlainText
            text: root.strings.t("now.error")
            color: root.urgent
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }

        Text {
            visible: root.vm.computeBusy && root.phase === "sleeping" && root.vm.images === "idle"
            width: parent.width
            wrapMode: Text.WordWrap
            textFormat: Text.PlainText
            text: root.strings.t("now.computeBusy")
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
    }
}
