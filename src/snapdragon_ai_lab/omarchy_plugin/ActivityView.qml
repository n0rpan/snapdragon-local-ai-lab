import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui

// Recent requests: four figures, then the latest requests with speed and context.
Column {
    id: root

    width: parent ? parent.width : 0
    spacing: Style.space(14)

    property var vm
    property var strings
    property color foreground
    property color accent
    property color urgent
    property string face
    property bool busy: false
    property var ledger

    component T: Label { color: root.foreground; font.family: root.face }

    readonly property bool available: !!(ledger && ledger.available)
    readonly property var rows: available && ledger.rows ? ledger.rows.slice(0, 8) : []
    readonly property var summary: ledger && ledger.summary ? ledger.summary : ({})
    readonly property real colWhen: Style.space(48)

    readonly property var statCells: [
        { label: word("activity.stat.requests"), value: fmt(summary.count || 0, 0) },
        { label: word("activity.stat.ttft"), value: secondsText(summary.medianTtftMs) },
        { label: word("activity.stat.speed"), value: speedText(summary.medianAnswerTokS) },
        { label: word("activity.stat.cache"), value: percentText(summary.cacheShare) }
    ]

    function word(key) { return strings ? strings.t(key) : "" }
    function fmt(value, decimals) { return Commons.I18n.formatNumber(value, decimals) }
    function isValue(v) { return typeof v === "number" && isFinite(v) && v >= 0 }
    function secondsText(ms) { return isValue(ms) ? word("unit.seconds").arg(fmt(ms / 1000, 1)) : word("unit.none") }
    function speedText(tokS) { return isValue(tokS) ? word("unit.toks").arg(fmt(tokS, tokS < 10 ? 1 : 0)) : word("unit.none") }
    function percentText(share) { return isValue(share) ? word("unit.percent").arg(fmt(Math.round(share * 100), 0)) : word("unit.none") }
    function agoText(t) {
        if (typeof t !== "number" || !isFinite(t)) return ""
        var age = Date.now() / 1000 - t
        if (age < 60) return word("activity.ago.now")
        if (age < 3600) return word("activity.ago.minutes").arg(fmt(Math.floor(age / 60), 0))
        if (age < 172800) return word("activity.ago.hours").arg(fmt(Math.floor(age / 3600), 0))
        return word("activity.ago.days").arg(fmt(Math.floor(age / 86400), 0))
    }
    function whoText(row) {
        var client = row.harness ? row.harness : word("activity.unknownClient")
        return row.model ? word("activity.row.who").arg(client).arg(word("model." + row.model + ".short")) : client
    }
    // The dim second line: first token, context size and cache share, tokens out, waits.
    function detailText(row) {
        var parts = []
        if (row.ok && isValue(row.ttftMs)) parts.push(word("activity.row.first").arg(fmt(row.ttftMs / 1000, 1)))
        if (isValue(row.cacheN) && isValue(row.promptN) && row.cacheN + row.promptN > 0) {
            var context = fmt(row.cacheN + row.promptN, 0)
            parts.push(isValue(row.cacheShare)
                ? word("activity.row.context").arg(context).arg(fmt(Math.round(row.cacheShare * 100), 0))
                : word("activity.row.contextOnly").arg(context))
        }
        // Where the reused prefix came from; only when the ledger says so.
        if (row.cacheSrc === "cold") parts.push(word("activity.row.src.cold"))
        else if (row.cacheSrc !== "" && isValue(row.cacheN) && row.cacheN > 0) parts.push(word("activity.row.src." + row.cacheSrc).arg(fmt(row.cacheN, 0)))
        if (row.missClass === "unknown") parts.push(word("activity.row.miss"))
        if (row.predictedN > 0) parts.push(word("activity.row.out").arg(fmt(row.predictedN, 0)))
        if (row.loadMs > 0) parts.push(word("activity.row.load").arg(fmt(row.loadMs / 1000, 1)))
        if (row.queueMs >= 500) parts.push(word("activity.row.queue").arg(fmt(row.queueMs / 1000, 1)))
        return parts.join(" · ")
    }

    T {
        visible: !root.available || root.rows.length === 0
        width: root.width
        wrapMode: Text.WordWrap
        text: root.word(root.available ? "activity.empty" : "activity.unavailable")
        dim: true
    }

    Row {
        id: statRow
        visible: root.rows.length > 0
        width: root.width
        spacing: Style.space(8)

        Repeater {
            model: root.statCells
            delegate: Column {
                required property var modelData
                width: (statRow.width - 3 * statRow.spacing) / 4
                spacing: Style.space(2)

                T {
                    width: parent.width
                    text: modelData.value
                    font.pixelSize: Style.font.title
                }
                T {
                    width: parent.width
                    text: modelData.label
                    dim: true
                    font.pixelSize: Style.font.caption
                }
            }
        }
    }

    Column {
        visible: root.rows.length > 0
        width: root.width
        spacing: Style.space(10)

        Repeater {
            model: root.rows

            delegate: Item {
                id: entry

                required property var modelData
                readonly property var row: modelData
                readonly property bool lab: row.origin === "lab"
                readonly property string detail: root.detailText(row)

                width: root.width
                height: body.height

                T {
                    id: when
                    width: root.colWhen
                    text: root.agoText(entry.row.t)
                    dim: true
                    font.pixelSize: Style.font.bodySmall
                    anchors.baseline: body.top
                    anchors.baselineOffset: who.baselineOffset
                }

                Column {
                    id: body
                    x: root.colWhen + Style.space(8)
                    width: entry.width - x
                    spacing: Style.space(2)

                    Item {
                        width: parent.width
                        height: who.implicitHeight

                        T {
                            id: who
                            width: Math.min(implicitWidth, parent.width - result.width - labTag.width - Style.space(16))
                            text: root.whoText(entry.row)
                        }
                        T {
                            id: labTag
                            anchors.left: who.right
                            anchors.leftMargin: Style.space(8)
                            anchors.baseline: who.baseline
                            visible: entry.lab
                            width: visible ? implicitWidth : 0
                            text: root.word("activity.row.lab")
                            dim: true
                            font.pixelSize: Style.font.caption
                        }
                        T {
                            id: result
                            anchors.right: parent.right
                            anchors.baseline: who.baseline
                            text: entry.row.ok ? root.speedText(entry.row.answerTokS)
                                : root.word("activity.row.failed").arg(root.fmt(Math.max(0, entry.row.status), 0))
                            color: entry.row.ok || entry.lab ? root.foreground : root.urgent
                            dim: entry.row.ok ? !(entry.row.answerTokS >= 0) : entry.lab
                        }
                    }
                    T {
                        visible: entry.detail !== ""
                        width: parent.width
                        text: entry.detail
                        wrapMode: Text.WordWrap
                        maximumLineCount: 2
                        dim: true
                        font.pixelSize: Style.font.caption
                    }
                }
            }
        }
    }
}
