import QtQuick
import qs.Commons
import qs.Commons as Commons

// One stacked bar: the loaded model, other apps, then free RAM. While the selected model
// is not loaded, a ghost segment shows the room it would take (needGib > 0), tinted
// urgent when it does not fit.
Item {
    id: root

    property real totalGib: -1
    property real modelGib: -1
    property real otherGib: -1
    property real needGib: -1
    property bool needFits: true
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property color urgent: Commons.Color.urgent

    readonly property bool known: totalGib > 0

    function share(gib) {
        return root.totalGib > 0 && gib > 0 ? Math.min(1, gib / root.totalGib) : 0
    }
    function tint(c, a) { return Qt.rgba(c.r, c.g, c.b, a) }

    readonly property real modelShare: share(modelGib)
    readonly property real otherShare: Math.min(1 - modelShare, share(otherGib))
    readonly property real needShare: Math.min(1 - modelShare - otherShare, share(needGib))
    readonly property real gap: Style.space(2)

    width: parent ? parent.width : implicitWidth
    implicitWidth: Style.space(200)
    implicitHeight: known ? Style.space(6) : 0
    height: implicitHeight
    visible: known

    Rectangle {
        id: track
        anchors.fill: parent
        radius: Style.cornerRadius > 0 ? height / 2 : 0
        color: root.tint(root.foreground, 0.10)
        clip: true

        Rectangle {
            id: modelSegment
            height: track.height
            width: Math.round(root.modelShare * track.width)
            color: root.accent
            Behavior on width { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
        }

        Rectangle {
            id: otherSegment
            x: modelSegment.width > 0 ? modelSegment.width + root.gap : 0
            height: track.height
            width: Math.round(root.otherShare * track.width)
            color: root.tint(root.foreground, 0.38)
            Behavior on width { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
        }

        Rectangle {
            visible: root.needGib > 0
            x: otherSegment.x + otherSegment.width + (otherSegment.width > 0 ? root.gap : 0)
            height: track.height
            width: Math.max(0, Math.min(track.width - x, Math.round(root.share(root.needGib) * track.width)))
            color: root.needFits ? root.tint(root.accent, 0.35) : root.tint(root.urgent, 0.55)
            Behavior on width { NumberAnimation { duration: Style.duration(180); easing.type: Easing.OutCubic } }
        }
    }
}
