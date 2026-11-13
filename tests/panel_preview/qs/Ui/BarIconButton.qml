import QtQuick
// Preview stand-in for the bar icon: properties only.
Item {
  property QtObject bar: null
  property string text: ""
  property string labelText: ""
  property string tooltipText: ""
  property real slotSize: 24
  signal pressed()
  implicitWidth: slotSize
  implicitHeight: slotSize
}
