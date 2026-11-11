import QtQuick
import qs.Commons
import qs.Commons as Commons
Text {
  property color foreground: Commons.Color.foreground
  property string fontFamily: Style.font.family
  property real fontSize: Style.font.caption
  textFormat: Text.PlainText
  color: foreground; opacity: 0.6
  font.family: fontFamily; font.pixelSize: fontSize; font.capitalization: Font.AllUppercase; font.letterSpacing: 1
}
