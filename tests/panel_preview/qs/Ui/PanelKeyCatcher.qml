import QtQuick
// Preview stand-in: the signals exist; no keys are handled offscreen.
Item {
  property bool blocked: false
  signal moveRequested(int dx, int dy)
  signal activateRequested()
  signal closeRequested()
  signal tabRequested(int direction)
}
