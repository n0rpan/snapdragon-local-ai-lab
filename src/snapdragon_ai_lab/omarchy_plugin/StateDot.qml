import QtQuick
import qs.Commons
import qs.Commons as Commons

// A small state mark: filled, hollow (outline) or pulsing. Square on square-cornered themes.
Item {
    id: mark

    property string kind: "filled"
    property color tint: Commons.Color.foreground

    width: Style.space(8)
    height: width

    Rectangle {
        anchors.fill: parent
        radius: Style.cornerRadius > 0 ? width / 2 : 0
        color: mark.kind === "hollow" ? "transparent" : mark.tint
        border.width: mark.kind === "hollow" ? 1 : 0
        border.color: Qt.rgba(mark.tint.r, mark.tint.g, mark.tint.b, 0.6)

        SequentialAnimation on opacity {
            running: mark.kind === "pulse" && !Style.reduceMotion
            loops: Animation.Infinite
            alwaysRunToEnd: true
            NumberAnimation { from: 1; to: 0.3; duration: Style.duration(900); easing.type: Easing.InOutQuad }
            NumberAnimation { from: 0.3; to: 1; duration: Style.duration(900); easing.type: Easing.InOutQuad }
        }
    }
}
