import QtQuick
import qs.Commons

// The panel's only text element: always plain text, body size, optionally dimmed.
Text {
    property bool dim: false

    textFormat: Text.PlainText
    font.pixelSize: Style.font.body
    opacity: dim ? 0.6 : 1
    elide: Text.ElideRight
}
