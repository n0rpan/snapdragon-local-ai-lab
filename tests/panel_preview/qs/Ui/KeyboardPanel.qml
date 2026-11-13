import QtQuick
import Quickshell
import qs.Commons as Commons
// Preview stand-in for the kit's popup: the content is built on first open (as in the kit) and
// drawn in place on the panel background.
Item {
  id: root
  property Item anchorItem: null
  property QtObject bar: null
  property var owner: null
  property bool open: false
  property Item focusTarget: null
  property int padding: 14
  property int contentWidth: 280
  property int contentHeight: 200
  property Component content: null
  readonly property Item loadedContent: loader.item
  readonly property real loadedContentHeight: loader.item ? loader.item.implicitHeight : 0
  function fittedContentWidth(width, cap) { return width }
  function fittedContentHeight(implicitHeight, cap) { return implicitHeight }
  width: contentWidth + padding * 2
  height: contentHeight + padding * 2
  visible: open
  onHeightChanged: Feed.cardHeight = height
  Rectangle { anchors.fill: parent; color: Commons.Color.background }
  Loader {
    id: loader
    x: root.padding; y: root.padding
    width: root.contentWidth; height: root.contentHeight
    active: root.content !== null && root.open
    sourceComponent: root.content
  }
}
