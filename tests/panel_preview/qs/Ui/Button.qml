import QtQuick
import qs.Commons
import qs.Commons as Commons
Rectangle {
  id: root
  property string text: ""
  property string iconText: ""
  property string tooltipText: ""
  property bool selected: false
  property bool bordered: false
  property bool focusable: false
  property color foreground: Commons.Color.foreground
  property color background: "transparent"
  property color accent: Commons.Color.accent
  property string fontFamily: Style.font.family
  property real fontSize: Style.font.body
  property real iconSize: Style.font.icon
  readonly property bool hot: false
  property real horizontalPadding: Style.spacing.controlPaddingX
  property real verticalPadding: Style.spacing.controlPaddingY
  signal clicked()
  signal rightClicked()
  implicitWidth: label.implicitWidth + horizontalPadding * 2
  implicitHeight: label.implicitHeight + verticalPadding * 2
  color: selected ? Qt.rgba(foreground.r, foreground.g, foreground.b, 0.18) : Qt.rgba(foreground.r, foreground.g, foreground.b, 0.04)
  border.width: bordered ? 1 : 0
  border.color: Qt.rgba(foreground.r, foreground.g, foreground.b, selected ? 1 : 0.4)
  opacity: enabled ? 1 : 0.45
  Text { id: label; anchors.centerIn: parent; textFormat: Text.PlainText; text: (root.iconText !== "" && root.text !== "" ? root.iconText + "  " : root.iconText) + root.text; color: root.foreground; font.family: root.fontFamily; font.pixelSize: root.fontSize }
  MouseArea { anchors.fill: parent; onClicked: root.clicked() }
}
