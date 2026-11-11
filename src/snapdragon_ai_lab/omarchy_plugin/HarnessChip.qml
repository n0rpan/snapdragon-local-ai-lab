import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui

// One harness: ready (filled dot), warm-up (ring, with Prepare) or preparing (pulse).
// Right click forgets the harness. The counters are kept for the detail text, which
// the kit tooltip cannot show here, so it is not drawn.
Rectangle {
    id: root

    property string name: ""
    property string markState: "ready"
    property int estimateSeconds: -1
    property int tools: 0
    property real cachedTokens: 0
    property real evaluatedTokens: 0
    property var strings: null
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property string face: Style.font.family
    property bool busy: false

    signal prepareRequested()
    signal forgetRequested()

    readonly property color ink: foreground
    readonly property color hot: accent

    readonly property string label: {
        if (root.markState === "preparing") return root.strings.t("harness.preparing").arg(root.name)
        if (root.markState === "warmup") {
            return root.estimateSeconds < 0
                ? root.strings.t("harness.warmupUnknown").arg(root.name)
                : root.strings.t("harness.warmup").arg(root.name).arg(Commons.I18n.formatNumber(root.estimateSeconds))
        }
        return root.strings.t("harness.ready").arg(root.name)
    }

    implicitWidth: row.implicitWidth + Style.space(10) * 2
    implicitHeight: Math.max(row.implicitHeight, Style.space(12)) + Style.space(6) * 2
    height: implicitHeight
    radius: height / 2
    color: "transparent"
    border.width: 1
    border.color: Qt.rgba(root.ink.r, root.ink.g, root.ink.b, 0.3)

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.RightButton
        onClicked: root.forgetRequested()
    }

    // Same dot as the other views, inline so this file stands alone.
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

    Row {
        id: row
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: Style.space(10)
        spacing: Style.space(8)

        Item {
            width: Style.space(8)
            height: labelText.implicitHeight

            StateMark {
                anchors.verticalCenter: parent.verticalCenter
                kind: root.markState === "preparing" ? "pulse" : root.markState === "warmup" ? "hollow" : "filled"
                tint: root.markState === "preparing" ? root.hot : root.ink
            }
        }

        Text {
            id: labelText
            textFormat: Text.PlainText
            text: root.label
            color: root.ink
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }

        Ui.Button {
            visible: root.markState === "warmup"
            text: root.strings.t("harness.prepare")
            enabled: !root.busy
            opacity: enabled ? 1 : 0.5
            bordered: true
            foreground: root.ink
            accent: root.hot
            fontFamily: root.face
            fontSize: Style.font.caption
            onClicked: root.prepareRequested()
        }
    }
}
