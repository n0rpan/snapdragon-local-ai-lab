import QtQuick
import qs.Commons
import qs.Commons as Commons
Rectangle {
  id: root
  property bool checked: false
  property bool busy: false
  property bool interactive: true
  property color foreground: Commons.Color.foreground
  property color accent: Commons.Color.accent
  signal toggled()
  implicitWidth: 28; implicitHeight: 14; radius: 7
  color: checked ? accent : Qt.rgba(foreground.r, foreground.g, foreground.b, 0.2)
  Rectangle { width: 10; height: 10; radius: 5; y: 2; x: root.checked ? 16 : 2; color: root.foreground }
  MouseArea { anchors.fill: parent; onClicked: root.toggled() }
}
