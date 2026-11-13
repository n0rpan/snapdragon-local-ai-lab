import QtQuick
// Preview stand-in: tooltips never show offscreen; only the properties exist. The marker lets
// the full-panel preview read a tooltip's text without hovering.
Item {
  property string text: ""
  property string fontFamily: ""
  property real fontSize: 11
  property int delay: 400
  readonly property bool previewToolTip: true
  visible: false
}
