import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui

// One client app (harness) as a quiet row: ready (filled dot), cold (outline, with a
// warm-up button) or warming up (pulse). Right click forgets it; the counters are in the tooltip.
Item {
    id: root

    property string name: ""
    property string markState: "ready"
    property int estimateSeconds: -1
    property int tools: 0
    property real cachedTokens: 0
    property real evaluatedTokens: 0
    property string costText: ""
    property var strings: null
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property string face: Style.font.family
    property bool busy: false

    signal prepareRequested()
    signal forgetRequested()

    function fmt(value) { return Commons.I18n.formatNumber(value) }

    readonly property string status: {
        if (root.markState === "preparing") return root.strings.t("harness.preparing")
        if (root.markState === "warmup") {
            return root.estimateSeconds < 0
                ? root.strings.t("harness.warmupUnknown")
                : root.strings.t("harness.warmup").arg(root.fmt(root.estimateSeconds))
        }
        return root.strings.t("harness.ready")
    }
    readonly property string tip: {
        var lines = [root.strings.t("harness.detail").arg(root.fmt(root.tools))
            .arg(root.fmt(root.cachedTokens)).arg(root.fmt(root.evaluatedTokens))]
        if (root.markState === "warmup" && root.costText !== "") lines.push(root.costText)
        lines.push(root.strings.t("harness.forget"))
        return lines.join("\n")
    }

    implicitHeight: Math.max(nameLabel.implicitHeight, prepareButton.implicitHeight)
    height: implicitHeight

    MouseArea {
        id: hover
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.RightButton
        onClicked: root.forgetRequested()
    }

    Ui.PanelToolTip {
        visible: hover.containsMouse
        text: root.tip
        width: Math.min(implicitWidth, Style.space(340))
        fontFamily: root.face
    }

    StateDot {
        id: dot
        anchors.verticalCenter: parent.verticalCenter
        kind: root.markState === "preparing" ? "pulse" : root.markState === "warmup" ? "hollow" : "filled"
        tint: root.markState === "preparing" ? root.accent : root.foreground
    }

    Label {
        id: nameLabel
        anchors.left: dot.right
        anchors.leftMargin: Style.space(10)
        anchors.verticalCenter: parent.verticalCenter
        width: Math.min(implicitWidth, root.width * 0.4)
        text: root.name
        color: root.foreground
        font.family: root.face
    }

    Label {
        anchors.left: nameLabel.right
        anchors.leftMargin: Style.space(10)
        anchors.right: prepareButton.visible ? prepareButton.left : parent.right
        anchors.rightMargin: prepareButton.visible ? Style.space(8) : 0
        anchors.verticalCenter: parent.verticalCenter
        text: root.status
        dim: true
        color: root.foreground
        font.family: root.face
        font.pixelSize: Style.font.bodySmall
    }

    Ui.Button {
        id: prepareButton
        visible: root.markState === "warmup"
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        text: root.strings.t("harness.prepare")
        enabled: !root.busy
        opacity: enabled ? 1 : 0.5
        bordered: true
        foreground: root.foreground
        accent: root.accent
        fontFamily: root.face
        fontSize: Style.font.bodySmall
        verticalPadding: Style.spacing.xs
        onClicked: root.prepareRequested()
    }
}
