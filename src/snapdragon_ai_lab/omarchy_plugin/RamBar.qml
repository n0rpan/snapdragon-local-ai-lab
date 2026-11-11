import QtQuick
import qs.Commons
import qs.Commons as Commons

// One stacked bar: the loaded model, other apps, then free RAM. The marker
// shows where the bar would end if the selected model loaded (needGib > 0).
Item {
    id: root

    property real totalGib: -1
    property real modelGib: -1
    property real otherGib: -1
    property real needGib: -1
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent

    readonly property bool known: totalGib > 0

    function share(gib) {
        return root.totalGib > 0 && gib > 0 ? Math.min(1, gib / root.totalGib) : 0
    }

    readonly property real modelShare: share(modelGib)
    readonly property real otherShare: Math.min(1 - modelShare, share(otherGib))
    readonly property real markerShare: Math.min(1, modelShare + otherShare + share(needGib))

    width: parent ? parent.width : implicitWidth
    implicitWidth: Style.space(200)
    implicitHeight: known ? Style.space(8) : 0
    height: implicitHeight

    Rectangle {
        id: track
        anchors.fill: parent
        radius: height / 2
        color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.10)
        clip: true

        Rectangle {
            id: modelSegment
            x: 0
            y: 0
            height: track.height
            width: root.modelShare * track.width
            color: root.accent
            Behavior on width { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
        }

        Rectangle {
            id: otherSegment
            x: modelSegment.width
            y: 0
            height: track.height
            width: root.otherShare * track.width
            color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.35)
            Behavior on width { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
            Behavior on x { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
        }

        Rectangle {
            visible: root.needGib > 0
            x: Math.min(track.width - width, root.markerShare * track.width)
            y: 0
            width: 1
            height: track.height
            color: root.foreground
        }
    }
}
