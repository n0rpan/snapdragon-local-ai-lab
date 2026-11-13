import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui
import "Model.js" as Model

// Sharing the models with other devices: off (the default), the tailnet (recommended) or the
// local network. Shown only when the control command reports the share state. What each choice
// exposes is in its tooltip; plain-HTTP LAN sharing says so in the open. The key itself is never
// shown here: only its fingerprint, and the command that reveals it once.
Column {
    id: root

    property var share: Model.parseShare("")
    property var strings: null
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property color urgent: Commons.Color.urgent
    property string face: Style.font.family
    property bool busy: false

    signal shareRequested(string mode)

    // Turning sharing on is confirmed in place first: `pending` is the mode waiting for a yes.
    // Turning it off happens at once.
    property string pending: ""
    function pick(mode) {
        if (root.busy) return
        if (mode === root.share.mode || mode === "off") {
            root.pending = ""
            if (mode !== root.share.mode) root.shareRequested("off")
        } else if (mode === "tailscale" || mode === "lan") {
            root.pending = mode
        }
    }
    function confirm() {
        if (root.pending === "" || root.busy) return
        var mode = root.pending
        root.pending = ""
        root.shareRequested(mode)
    }
    function cancel() { root.pending = "" }

    objectName: "localAiShare"

    readonly property bool on: root.share.enabled === true
    // One line under the choice, only while sharing is on (or was asked for and is not up).
    readonly property string stateText: !root.on ? ""
        : root.share.active && root.share.urls.length > 0 ? root.strings.t("share.active").arg(root.share.urls[0])
        : root.share.active ? root.strings.t("share.activePlain")
        : root.strings.t("share.inactive")
    readonly property string stateTip: {
        if (!root.on) return ""
        var lines = root.share.urls.slice(1)
        if (root.share.error !== "") lines.push(root.share.error)
        if (root.share.fingerprint !== "") lines.push(root.strings.t("share.key").arg(root.share.fingerprint))
        if (root.share.remoteRefused > 0)
            lines.push(root.strings.tn("share.refused", root.share.remoteRefused).arg(Commons.I18n.formatNumber(root.share.remoteRefused, 0)))
        return lines.join("\n")
    }
    // Plain HTTP on the local network: said in the open, not in a tooltip.
    readonly property bool plain: root.on && root.share.active === true && root.share.mode === "lan" && root.share.encryption !== "tls"
    // Use by other devices, only once there is some.
    readonly property string remoteText: {
        if (!root.on || root.share.remoteRequests <= 0) return ""
        var text = root.strings.tn("share.remote", root.share.remoteRequests).arg(Commons.I18n.formatNumber(root.share.remoteRequests, 0))
        return root.share.remoteActive > 0 ? text + " · " + root.strings.t("share.remoteNow") : text
    }

    visible: root.share.available === true
    spacing: Style.space(6)

    Ui.PanelSectionHeader {
        foreground: root.foreground
        fontFamily: root.face
        text: root.strings.t("share.title")
    }

    Ui.ButtonGroup {
        enabled: !root.busy
        opacity: enabled ? 1 : 0.5
        options: [
            { value: "off", label: root.strings.t("share.off"), tooltip: root.strings.t("share.off.tip") },
            { value: "tailscale", label: root.strings.t("share.tailscale"), tooltip: root.strings.t("share.tailscale.tip") },
            { value: "lan", label: root.strings.t("share.lan"), tooltip: root.strings.t("share.lan.tip") }
        ]
        value: root.share.mode
        foreground: root.foreground
        accent: root.accent
        fontFamily: root.face
        fontSize: Style.font.bodySmall
        focusable: false
        onChanged: function(picked) { root.pick(picked) }
    }

    // What turning it on shares, with whom, and what protects it; then one yes or no.
    Column {
        visible: root.pending !== ""
        width: root.width
        spacing: Style.space(6)

        Label {
            width: parent.width
            text: root.pending !== "" ? root.strings.t("share.confirm." + root.pending) : ""
            color: root.foreground
            wrapMode: Text.WordWrap
            maximumLineCount: 6
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
        Row {
            spacing: Style.space(6)

            Ui.Button {
                text: root.strings.t("share.confirm.yes")
                bordered: true
                enabled: !root.busy
                foreground: root.accent
                accent: root.accent
                fontFamily: root.face
                fontSize: Style.font.bodySmall
                onClicked: root.confirm()
            }
            Ui.Button {
                text: root.strings.t("clear.cancel")
                foreground: root.foreground
                accent: root.accent
                fontFamily: root.face
                fontSize: Style.font.bodySmall
                onClicked: root.cancel()
            }
        }
    }

    Label {
        visible: root.stateText !== ""
        width: root.width
        text: root.stateText
        color: root.on && !root.share.active ? root.urgent : root.foreground
        dim: root.share.active === true
        wrapMode: Text.WordWrap
        maximumLineCount: 2
        font.family: root.face
        font.pixelSize: Style.font.bodySmall

        MouseArea { id: stateHover; anchors.fill: parent; hoverEnabled: true; enabled: root.stateTip !== "" }
        Ui.PanelToolTip {
            visible: stateHover.enabled && stateHover.containsMouse
            text: root.stateTip
            width: Math.min(implicitWidth, Style.space(340))
            fontFamily: root.face
        }
    }

    Label {
        visible: root.plain
        width: root.width
        text: root.strings.t("share.lanWarning")
        color: root.urgent
        wrapMode: Text.WordWrap
        maximumLineCount: 3
        font.family: root.face
        font.pixelSize: Style.font.bodySmall
    }

    Label {
        visible: root.remoteText !== ""
        width: root.width
        text: root.remoteText
        color: root.foreground
        dim: true
        font.family: root.face
        font.pixelSize: Style.font.caption
    }
}
