import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui as Ui
import "Model.js" as Model

// The model chooser (named, then heavy), the flash download state and the harness strip.
Column {
    id: root

    property var vm: Model.parse("", 0)
    property var strings: null
    property color foreground: Commons.Color.foreground
    property color accent: Commons.Color.accent
    property color urgent: Commons.Color.urgent
    property string face: Style.font.family
    property bool busy: false
    property var preparing: []

    signal selectRequested(string key)
    signal prepareRequested(string name)
    signal prepareAllRequested()
    signal forgetRequested(string name)

    function fmt(value, decimals) {
        return Commons.I18n.formatNumber(value, decimals)
    }

    // A dot that is filled, hollow (ring) or pulsing.
    component StateMark: Item {
        id: mark
        property string kind: "filled"
        property color tint: Commons.Color.foreground
        width: Style.space(8)
        height: width

        Rectangle {
            anchors.fill: parent
            radius: width / 2
            color: mark.kind === "hollow" ? "transparent" : mark.tint
            border.width: mark.kind === "hollow" ? 1 : 0
            border.color: mark.tint

            SequentialAnimation on opacity {
                running: mark.kind === "pulse" && !Style.reduceMotion
                loops: Animation.Infinite
                NumberAnimation { from: 1; to: 0.3; duration: Style.duration(900); easing.type: Easing.InOutQuad }
                NumberAnimation { from: 0.3; to: 1; duration: Style.duration(900); easing.type: Easing.InOutQuad }
            }
        }
    }

    // One chooser row. Clicking it asks to select the model; the parent decides nothing.
    component ModelRow: Item {
        id: entry
        property var info: ({})
        readonly property string key: entry.info.key || ""
        readonly property bool selected: entry.info.selected === true
        readonly property bool available: entry.info.available === true
        readonly property bool clickable: entry.available && !entry.selected && !root.busy && root.canSelect
        readonly property bool needKnown: (entry.info.needGib || -1) >= 0
        readonly property bool noFit: entry.available && entry.info.fits === "no" && entry.needKnown
        readonly property real pad: Style.space(8)
        readonly property real gap: Style.space(10)
        readonly property real radioSize: Style.space(14)
        readonly property string badgeText: entry.info.isDefault === true ? root.strings.t("model.badge.default")
            : entry.info.group === "heavy" ? root.strings.t("model.badge.heavy") : ""

        width: root.width
        height: body.height + entry.pad * 2
        opacity: entry.available ? 1 : 0.5

        Rectangle {
            anchors.fill: parent
            radius: Style.cornerRadius
            color: hover.containsMouse && entry.clickable
                ? Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.06) : "transparent"
        }

        MouseArea {
            id: hover
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: entry.clickable ? Qt.PointingHandCursor : Qt.ArrowCursor
            onClicked: {
                if (entry.clickable) root.selectRequested(entry.key)
            }
        }

        Item {
            id: body
            x: entry.pad
            y: entry.pad
            width: entry.width - entry.pad * 2
            height: nameText.implicitHeight + Style.space(2) + detailCol.height

            Rectangle {
                id: radio
                x: 0
                y: (nameText.implicitHeight - height) / 2
                width: entry.radioSize
                height: width
                radius: width / 2
                color: "transparent"
                border.width: 1
                border.color: root.foreground

                Rectangle {
                    anchors.centerIn: parent
                    width: parent.width / 2
                    height: width
                    radius: width / 2
                    color: root.accent
                    visible: entry.selected
                }
            }

            Text {
                id: nameText
                x: radio.width + entry.gap
                textFormat: Text.PlainText
                text: root.strings.t("model." + entry.key)
                color: root.foreground
                font.family: root.face
                font.pixelSize: Style.font.subtitle
            }

            Rectangle {
                id: badge
                anchors.left: nameText.right
                anchors.leftMargin: Style.space(8)
                anchors.verticalCenter: nameText.verticalCenter
                visible: entry.badgeText !== ""
                width: badgeLabel.implicitWidth + Style.space(12)
                height: badgeLabel.implicitHeight + Style.space(2)
                radius: height / 2
                color: "transparent"
                border.width: 1
                border.color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.4)

                Text {
                    id: badgeLabel
                    anchors.centerIn: parent
                    textFormat: Text.PlainText
                    text: entry.badgeText
                    color: root.foreground
                    font.family: root.face
                    font.pixelSize: Style.font.caption
                }
            }

            Item {
                id: stateGroup
                anchors.right: parent.right
                anchors.verticalCenter: nameText.verticalCenter
                width: stateMark.width + Style.space(6) + stateText.implicitWidth
                height: stateText.implicitHeight

                StateMark {
                    id: stateMark
                    anchors.verticalCenter: parent.verticalCenter
                    kind: entry.info.state === "loaded" ? "filled"
                        : entry.info.state === "loading" ? "pulse" : "hollow"
                    tint: entry.info.state === "cold" ? root.foreground : root.accent
                }

                Text {
                    id: stateText
                    anchors.left: stateMark.right
                    anchors.leftMargin: Style.space(6)
                    anchors.verticalCenter: parent.verticalCenter
                    textFormat: Text.PlainText
                    text: root.strings.t("model.state." + (entry.info.state || "cold"))
                    color: root.foreground
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }
            }

            Column {
                id: detailCol
                x: radio.width + entry.gap
                y: nameText.implicitHeight + Style.space(2)
                width: body.width - x
                spacing: Style.space(2)

                Text {
                    visible: entry.key === "flash"
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: root.strings.t("model.flash.full")
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: entry.available ? root.strings.t("model." + entry.key + ".hint") : root.strings.t("model.unavailable")
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    visible: entry.noFit
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: entry.noFit
                        ? root.strings.t("model.nofit").arg(root.fmt(entry.info.needGib, 1)).arg(root.fmt(root.vm.ram.availableGib, 1))
                        : ""
                    color: root.urgent
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    visible: entry.available && entry.needKnown && !entry.noFit
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: visible ? root.strings.t("model.need").arg(root.fmt(entry.info.needGib, 1)) : ""
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }

                Text {
                    visible: entry.available && (entry.info.batteryStart || 0) > 0
                    width: parent.width
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: visible ? root.strings.t("model.battery").arg(root.fmt(entry.info.batteryStart)) : ""
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.bodySmall
                }
            }
        }
    }

    readonly property var modelRows: root.vm.models || []
    readonly property var namedRows: root.modelRows.filter(function(m) { return m.group === "named" })
    readonly property var heavyRows: root.modelRows.filter(function(m) { return m.group === "heavy" })
    readonly property var harnessRows: root.vm.harnesses || []
    readonly property bool canSelect: root.vm.phase !== "loading" && root.vm.phase !== "answering"
    readonly property bool showHarnesses: root.vm.hasHarnessData === true && root.harnessRows.length > 0
    readonly property bool anyWarm: root.harnessRows.some(function(h) { return h.state === "warmup" })
    readonly property bool anyPreparing: root.preparing.length > 0
    readonly property string selectedName: root.vm.selected !== "" ? root.strings.t("model." + root.vm.selected + ".short") : ""

    readonly property string flashStateKey: {
        var s = root.vm.flash.state
        if (s === "downloading") return "flash.downloading"
        if (s === "verifying") return root.vm.flash.waiting ? "flash.waitingVerify" : "flash.verifying"
        if (s === "weights-ready-testing") return "flash.readyTesting"
        if (s === "paused") return "flash.paused"
        return ""
    }

    spacing: Style.space(8)

    Ui.PanelSectionHeader {
        width: root.width
        foreground: root.foreground
        fontFamily: root.face
        text: root.strings.t("model.group.named")
    }

    Column {
        width: root.width
        Repeater {
            model: root.namedRows
            delegate: ModelRow {
                required property var modelData
                info: modelData
            }
        }
    }

    Ui.PanelSectionHeader {
        width: root.width
        visible: root.heavyRows.length > 0
        foreground: root.foreground
        fontFamily: root.face
        text: root.strings.t("model.group.heavy")
    }

    Column {
        width: root.width
        Repeater {
            model: root.heavyRows
            delegate: ModelRow {
                required property var modelData
                info: modelData
            }
        }
    }

    Text {
        width: root.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: root.strings.t("model.selectHint")
        color: root.foreground
        opacity: 0.6
        font.family: root.face
        font.pixelSize: Style.font.caption
    }

    Column {
        visible: root.vm.flash.state !== "absent" || root.vm.flash.trialActive
        width: root.width
        spacing: Style.space(4)

        Text {
            visible: root.flashStateKey !== ""
            width: parent.width
            wrapMode: Text.WordWrap
            textFormat: Text.PlainText
            text: root.flashStateKey !== "" ? root.strings.t(root.flashStateKey) : ""
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }

        Text {
            visible: root.vm.flash.state !== "absent"
            width: parent.width
            wrapMode: Text.WordWrap
            textFormat: Text.PlainText
            text: visible
                ? root.strings.t("flash.progress")
                    .arg(root.fmt(root.vm.flash.percent, 1))
                    .arg(root.fmt(Model.gib(root.vm.flash.doneBytes), 1))
                    .arg(root.fmt(Model.gib(root.vm.flash.totalBytes), 1))
                    .arg(root.fmt(root.vm.flash.verified))
                    .arg(root.fmt(root.vm.flash.shards))
                : ""
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }

        Text {
            visible: root.vm.flash.trialActive === true
            width: parent.width
            wrapMode: Text.WordWrap
            textFormat: Text.PlainText
            text: visible
                ? root.strings.t("flash.trial").arg(root.fmt(Math.floor(root.vm.flash.trialElapsed / 60)))
                : ""
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
    }

    Item {
        visible: root.showHarnesses
        width: root.width
        height: Math.max(harnessHeader.implicitHeight, prepareAll.implicitHeight)

        Ui.PanelSectionHeader {
            id: harnessHeader
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            foreground: root.foreground
            fontFamily: root.face
            text: root.strings.t("harness.title")
        }

        Ui.Button {
            id: prepareAll
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            visible: root.anyWarm && !root.anyPreparing
            text: root.strings.t("harness.prepareAll")
            bordered: true
            foreground: root.foreground
            accent: root.accent
            fontFamily: root.face
            fontSize: Style.font.caption
            onClicked: root.prepareAllRequested()
        }
    }

    Flow {
        visible: root.showHarnesses
        width: root.width
        spacing: Style.space(6)

        Repeater {
            model: root.harnessRows
            delegate: HarnessChip {
                required property var modelData
                name: modelData.name
                markState: root.preparing.indexOf(modelData.name) >= 0 ? "preparing" : modelData.state
                estimateSeconds: modelData.estimateSeconds
                tools: modelData.tools
                cachedTokens: modelData.cachedTokens
                evaluatedTokens: modelData.evaluatedTokens
                strings: root.strings
                foreground: root.foreground
                accent: root.accent
                face: root.face
                busy: root.busy
                onPrepareRequested: root.prepareRequested(modelData.name)
                onForgetRequested: root.forgetRequested(modelData.name)
            }
        }
    }

    Text {
        visible: root.showHarnesses && root.anyWarm && root.vm.selected !== ""
        width: root.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: visible ? root.strings.t("harness.prepareCost").arg(root.selectedName) : ""
        color: root.foreground
        opacity: 0.6
        font.family: root.face
        font.pixelSize: Style.font.caption
    }
}
