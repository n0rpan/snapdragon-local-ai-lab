import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui

Column {
    id: root

    width: parent ? parent.width : 0
    spacing: Style.space(8)

    property var vm
    property var strings
    property color foreground
    property color accent
    property color urgent
    property string face
    property bool busy: false
    property var ledger

    property int expandedIndex: -1

    readonly property bool available: !!(ledger && ledger.available)
    readonly property var rows: available && ledger.rows ? ledger.rows.slice(0, 12) : []
    readonly property var summary: ledger && ledger.summary ? ledger.summary : ({})
    readonly property real lineHeight: Math.round(Style.font.body * 1.6)
    readonly property real colGap: Style.space(6)
    readonly property real colWhen: Style.space(44)
    readonly property real colInitial: Style.space(14)
    readonly property real colFirst: Style.space(80)
    readonly property real colSpeed: Style.space(72)
    readonly property real colCache: Style.space(48)
    readonly property real colClient: Math.max(Style.space(72), width - colWhen - colFirst - colSpeed - colCache - 4 * colGap)

    readonly property var statCells: [
        { label: tr("activity.stat.requests"), value: fmt(summary.count, 0) },
        { label: tr("activity.stat.ttft"), value: secondsText(summary.medianTtftMs) },
        { label: tr("activity.stat.speed"), value: speedText(summary.medianTokS, 1) },
        { label: tr("activity.stat.cache"), value: percentText(summary.cacheShare) }
    ]

    function tr(key) { return strings ? strings.t(key) : "" }
    function fmt(value, decimals) { return Commons.I18n.formatNumber(value, decimals) }
    function isValue(v) { return typeof v === "number" && isFinite(v) && v >= 0 }
    function secondsText(ms) { return isValue(ms) ? tr("unit.seconds").arg(fmt(ms / 1000, 1)) : tr("unit.none") }
    function speedText(tokS, decimals) { return isValue(tokS) ? tr("unit.toks").arg(fmt(tokS, decimals)) : tr("unit.none") }
    function percentText(share) { return isValue(share) ? tr("unit.percent").arg(fmt(Math.round(share * 100), 0)) : tr("unit.none") }
    function agoText(t) {
        if (typeof t !== "number" || !isFinite(t)) return ""
        var age = Date.now() / 1000 - t
        if (age < 60) return tr("activity.ago.now")
        if (age < 3600) return tr("activity.ago.minutes").arg(fmt(Math.floor(age / 60), 0))
        return tr("activity.ago.hours").arg(fmt(Math.floor(age / 3600), 0))
    }
    function modelInitial(key) { return key ? tr("model." + key + ".short").charAt(0) : "" }

    Text {
        visible: !root.available
        width: root.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: root.tr("activity.unavailable")
        color: root.foreground
        opacity: 0.6
        font.family: root.face
        font.pixelSize: Style.font.body
    }

    Text {
        visible: root.available && root.rows.length === 0
        width: root.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: root.tr("activity.empty")
        color: root.foreground
        opacity: 0.6
        font.family: root.face
        font.pixelSize: Style.font.body
    }

    Column {
        visible: root.available && root.rows.length > 0
        width: root.width
        spacing: Style.space(8)

        Row {
            id: statRow
            width: parent.width
            spacing: root.colGap

            Repeater {
                model: root.statCells
                delegate: Column {
                    width: (statRow.width - 3 * statRow.spacing) / 4
                    spacing: Style.space(2)

                    Text {
                        width: parent.width
                        elide: Text.ElideRight
                        textFormat: Text.PlainText
                        text: modelData.label
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                    Text {
                        width: parent.width
                        elide: Text.ElideRight
                        textFormat: Text.PlainText
                        text: modelData.value
                        color: root.foreground
                        font.family: root.face
                        font.pixelSize: Style.font.title
                    }
                }
            }
        }

        Ui.PanelSeparator { foreground: root.foreground }

        Row {
            width: parent.width
            spacing: root.colGap

            Text {
                width: root.colWhen
                height: root.lineHeight
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                textFormat: Text.PlainText
                text: root.tr("activity.col.when")
                color: root.foreground
                opacity: 0.6
                font.family: root.face
                font.pixelSize: Style.font.caption
            }
            Text {
                width: root.colClient
                height: root.lineHeight
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                textFormat: Text.PlainText
                text: root.tr("activity.col.client")
                color: root.foreground
                opacity: 0.6
                font.family: root.face
                font.pixelSize: Style.font.caption
            }
            Text {
                width: root.colFirst
                height: root.lineHeight
                horizontalAlignment: Text.AlignRight
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                textFormat: Text.PlainText
                text: root.tr("activity.col.first")
                color: root.foreground
                opacity: 0.6
                font.family: root.face
                font.pixelSize: Style.font.caption
            }
            Text {
                width: root.colSpeed
                height: root.lineHeight
                horizontalAlignment: Text.AlignRight
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                textFormat: Text.PlainText
                text: root.tr("activity.col.speed")
                color: root.foreground
                opacity: 0.6
                font.family: root.face
                font.pixelSize: Style.font.caption
            }
            Text {
                width: root.colCache
                height: root.lineHeight
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
                textFormat: Text.PlainText
                text: root.tr("activity.col.cache")
                color: root.foreground
                opacity: 0.6
                font.family: root.face
                font.pixelSize: Style.font.caption
            }
        }

        Repeater {
            model: root.rows

            delegate: Column {
                id: entry

                readonly property var row: modelData
                readonly property bool open: root.expandedIndex === index

                width: root.width
                spacing: Style.space(2)

                Item {
                    width: parent.width
                    height: line.height

                    Row {
                        id: line
                        width: parent.width
                        spacing: root.colGap

                        Text {
                            width: root.colWhen
                            height: root.lineHeight
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                            textFormat: Text.PlainText
                            text: root.agoText(entry.row.t)
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                        Row {
                            width: root.colClient
                            height: root.lineHeight
                            spacing: root.colGap

                            Text {
                                width: root.colClient - root.colInitial - root.colGap
                                height: root.lineHeight
                                verticalAlignment: Text.AlignVCenter
                                elide: Text.ElideRight
                                textFormat: Text.PlainText
                                text: entry.row.harness ? entry.row.harness : root.tr("activity.unknownClient")
                                color: root.foreground
                                font.family: root.face
                                font.pixelSize: Style.font.body
                            }
                            Text {
                                width: root.colInitial
                                height: root.lineHeight
                                verticalAlignment: Text.AlignVCenter
                                textFormat: Text.PlainText
                                text: root.modelInitial(entry.row.model)
                                color: root.foreground
                                font.family: root.face
                                font.pixelSize: Style.font.body
                            }
                        }
                        Text {
                            visible: entry.row.ok
                            width: root.colFirst
                            height: root.lineHeight
                            horizontalAlignment: Text.AlignRight
                            verticalAlignment: Text.AlignVCenter
                            textFormat: Text.PlainText
                            text: root.secondsText(entry.row.ttftMs)
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                        Text {
                            visible: entry.row.ok
                            width: root.colSpeed
                            height: root.lineHeight
                            horizontalAlignment: Text.AlignRight
                            verticalAlignment: Text.AlignVCenter
                            textFormat: Text.PlainText
                            text: root.speedText(entry.row.tokS, 0)
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                        Item {
                            visible: entry.row.ok
                            width: root.colCache
                            height: root.lineHeight

                            Rectangle {
                                visible: root.isValue(entry.row.cacheShare)
                                anchors.verticalCenter: parent.verticalCenter
                                width: parent.width
                                height: Style.space(6)
                                radius: height / 2
                                color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.10)

                                Rectangle {
                                    width: parent.width * Math.min(1, entry.row.cacheShare)
                                    height: parent.height
                                    radius: height / 2
                                    color: root.accent
                                }
                            }
                        }
                        Text {
                            visible: !entry.row.ok
                            width: root.colFirst + root.colSpeed + root.colCache + 2 * root.colGap
                            height: root.lineHeight
                            verticalAlignment: Text.AlignVCenter
                            elide: Text.ElideRight
                            textFormat: Text.PlainText
                            text: root.tr("activity.row.failed").arg(root.fmt(entry.row.status, 0))
                            color: root.urgent
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                    }

                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: root.expandedIndex = entry.open ? -1 : index
                    }
                }

                Item {
                    visible: entry.row.loadMs > 0 || entry.row.queueMs >= 500
                    width: parent.width
                    height: second.height

                    Row {
                        id: second
                        anchors.left: parent.left
                        anchors.leftMargin: root.colWhen + root.colGap
                        spacing: Style.space(8)

                        Text {
                            visible: entry.row.loadMs > 0
                            textFormat: Text.PlainText
                            text: root.tr("activity.row.load").arg(root.fmt(entry.row.loadMs / 1000, 1))
                            color: root.foreground
                            opacity: 0.6
                            font.family: root.face
                            font.pixelSize: Style.font.caption
                        }
                        Text {
                            visible: entry.row.queueMs >= 500
                            textFormat: Text.PlainText
                            text: root.tr("activity.row.queue").arg(root.fmt(entry.row.queueMs / 1000, 1))
                            color: root.foreground
                            opacity: 0.6
                            font.family: root.face
                            font.pixelSize: Style.font.caption
                        }
                    }
                }

                Item {
                    visible: entry.open
                    width: parent.width
                    height: tokens.height

                    Text {
                        id: tokens
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: root.colWhen + root.colGap
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                        text: root.tr("activity.row.tokens")
                            .arg(root.fmt(Math.max(0, entry.row.predictedN), 0))
                            .arg(root.fmt(Math.max(0, entry.row.cacheN), 0))
                            .arg(root.fmt(Math.max(0, entry.row.cacheN) + Math.max(0, entry.row.promptN), 0))
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                }
            }
        }
    }
}
