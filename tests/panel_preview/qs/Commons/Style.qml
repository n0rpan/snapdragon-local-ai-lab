pragma Singleton
import QtQuick
// Preview stand-in for Omarchy's Style: default token values only.
QtObject {
  property int cornerRadius: 0
  // No pulsing offscreen, so a preview image does not depend on when it is grabbed.
  readonly property bool reduceMotion: true
  readonly property QtObject bar: QtObject {
    readonly property int statusSlot: 24
  }
  readonly property var easeOut: [0.23, 1.0, 0.32, 1.0, 1.0, 1.0]
  function space(px) { return px }
  function duration(ms) { return ms }
  readonly property QtObject font: QtObject {
    readonly property string family: "JetBrainsMono Nerd Font"
    readonly property int caption: 10
    readonly property int bodySmall: 11
    readonly property int body: 12
    readonly property int subtitle: 13
    readonly property int title: 14
    readonly property int heading: 16
    readonly property int display: 24
    readonly property int icon: 14
  }
  readonly property QtObject spacing: QtObject {
    readonly property int xs: 3
    readonly property int sm: 4
    readonly property int md: 6
    readonly property int lg: 8
    readonly property int xl: 10
    readonly property int controlGap: 8
    readonly property int controlPaddingX: 10
    readonly property int controlPaddingY: 4
  }
}
