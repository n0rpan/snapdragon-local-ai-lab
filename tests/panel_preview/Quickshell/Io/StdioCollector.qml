import QtQuick
// Preview stand-in: holds the text the stand-in Process hands over.
QtObject {
  property bool waitForEnd: false
  property string text: ""
  signal streamFinished()
  function finish(value) { text = value; streamFinished() }
}
