import QtQuick
import qs.Commons
import qs.Commons as Commons
Row {
  id: root
  property var options: []
  property string value: ""
  property color foreground: Commons.Color.foreground
  property color background: Commons.Color.background
  property color accent: Commons.Color.accent
  property string fontFamily: Style.font.family
  property real fontSize: Style.font.body
  property bool focusable: true
  signal changed(string value)
  Repeater {
    model: root.options
    delegate: Button {
      required property var modelData
      text: modelData.label; selected: modelData.value === root.value; bordered: true
      tooltipText: modelData.tooltip ? String(modelData.tooltip) : ""  // as the kit does
      foreground: root.foreground; fontFamily: root.fontFamily; fontSize: root.fontSize
      onClicked: root.changed(modelData.value)
    }
  }
}
