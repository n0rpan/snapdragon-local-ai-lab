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
    property var storage

    signal deleteRequested(string id)
    signal clearRequested(bool prompts, bool conversations, bool harnesses)

    property string confirmId: ""
    property bool sheetOpen: false
    property bool clearPrompts: true
    property bool clearConversations: false
    property bool clearHarnesses: false

    readonly property bool available: !!(storage && storage.available)
    readonly property var entries: storage && storage.entries ? storage.entries : []
    readonly property bool hasHarness: !!(vm && vm.hasHarnessData)
    readonly property bool canConfirm: (clearPrompts || clearConversations || (clearHarnesses && hasHarness)) && !busy && !!vm && vm.active === 0
    readonly property real colGap: Style.space(6)
    readonly property real colSize: Style.space(72)
    readonly property real sheetPad: Style.space(10)
    readonly property var groups: {
        var order = ["ornith", "moe", "flash", "dense", "other"];
        var out = [];
        for (var i = 0; i < order.length; i++) {
            var items = [];
            for (var j = 0; j < entries.length; j++) {
                if (groupKey(entries[j].model) === order[i]) items.push(entries[j]);
            }
            if (items.length > 0) out.push({ key: order[i], items: items });
        }
        return out;
    }

    function tr(key) { return strings ? strings.t(key) : "" }
    function fmt(value, decimals) { return Commons.I18n.formatNumber(value, decimals) }
    function groupKey(modelKey) { return ["ornith", "moe", "flash", "dense"].indexOf(modelKey) >= 0 ? modelKey : "other" }
    function groupTitle(key) { return key === "other" ? tr("activity.unknownClient") : tr("model." + key + ".short") }
    function sizeText(bytes) {
        var n = Number(bytes) || 0;
        var gib = n / 1073741824;
        if (gib >= 0.1) return tr("unit.gib").arg(fmt(gib, 1));
        return tr("unit.mib").arg(fmt(n / 1048576, 0));
    }
    function closeSheet() {
        clearPrompts = true;
        clearConversations = false;
        clearHarnesses = false;
        sheetOpen = false;
    }

    Text {
        visible: !root.available
        width: root.width
        wrapMode: Text.WordWrap
        textFormat: Text.PlainText
        text: root.tr("storage.unavailable")
        color: root.foreground
        opacity: 0.6
        font.family: root.face
        font.pixelSize: Style.font.body
    }

    Flow {
        visible: root.available
        width: root.width
        spacing: Style.space(6)

        Text {
            textFormat: Text.PlainText
            text: root.tr("storage.total").arg(root.fmt((Number(root.storage ? root.storage.totalBytes : 0) || 0) / 1073741824, 1))
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
        Text {
            visible: (Number(root.storage ? root.storage.freeableBytes : 0) || 0) > 0
            textFormat: Text.PlainText
            text: "·"
            color: root.foreground
            opacity: 0.6
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
        Text {
            visible: (Number(root.storage ? root.storage.freeableBytes : 0) || 0) > 0
            textFormat: Text.PlainText
            text: root.tr("storage.freeable").arg(root.fmt((Number(root.storage ? root.storage.freeableBytes : 0) || 0) / 1073741824, 1))
            color: root.foreground
            font.family: root.face
            font.pixelSize: Style.font.bodySmall
        }
    }

    Repeater {
        model: root.available ? root.groups : []

        delegate: Column {
            id: groupBox

            readonly property var group: modelData

            width: root.width
            spacing: Style.space(4)

            Ui.PanelSectionHeader {
                foreground: root.foreground
                fontFamily: root.face
                text: root.groupTitle(groupBox.group.key)
            }

            Repeater {
                model: groupBox.group.items

                delegate: Column {
                    id: entryBox

                    readonly property var item: modelData
                    readonly property bool confirming: root.confirmId !== "" && root.confirmId === item.id

                    width: groupBox.width
                    spacing: Style.space(4)

                    Row {
                        width: parent.width
                        spacing: root.colGap

                        Text {
                            width: Math.max(0, parent.width - stateLabel.implicitWidth - sizeLabel.width - (deleteButton.visible ? deleteButton.width : 0) - 3 * parent.spacing)
                            elide: Text.ElideRight
                            textFormat: Text.PlainText
                            text: root.tr("storage.kind." + entryBox.item.kind)
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                        Text {
                            id: stateLabel
                            textFormat: Text.PlainText
                            text: root.tr("storage.state." + entryBox.item.state)
                            color: root.foreground
                            opacity: 0.6
                            font.family: root.face
                            font.pixelSize: Style.font.caption
                        }
                        Text {
                            id: sizeLabel
                            width: root.colSize
                            horizontalAlignment: Text.AlignRight
                            textFormat: Text.PlainText
                            text: root.sizeText(entryBox.item.bytes)
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.body
                        }
                        Ui.Button {
                            id: deleteButton
                            visible: !!entryBox.item.deletable && !entryBox.confirming
                            text: root.tr("storage.delete")
                            fontSize: Style.font.caption
                            bordered: true
                            foreground: root.foreground
                            fontFamily: root.face
                            enabled: !root.busy
                            onClicked: root.confirmId = entryBox.item.id
                        }
                    }

                    Row {
                        visible: entryBox.confirming
                        width: parent.width
                        spacing: root.colGap

                        Text {
                            width: Math.max(0, parent.width - confirmButton.width - cancelButton.width - 2 * parent.spacing)
                            wrapMode: Text.WordWrap
                            textFormat: Text.PlainText
                            text: root.tr("storage.deleteConfirm")
                            color: root.foreground
                            font.family: root.face
                            font.pixelSize: Style.font.bodySmall
                        }
                        Ui.Button {
                            id: confirmButton
                            text: root.tr("storage.delete")
                            fontSize: Style.font.caption
                            bordered: true
                            foreground: root.urgent
                            fontFamily: root.face
                            enabled: !root.busy
                            onClicked: {
                                root.deleteRequested(entryBox.item.id);
                                root.confirmId = "";
                            }
                        }
                        Ui.Button {
                            id: cancelButton
                            text: root.tr("clear.cancel")
                            fontSize: Style.font.caption
                            bordered: true
                            foreground: root.foreground
                            fontFamily: root.face
                            onClicked: root.confirmId = ""
                        }
                    }
                }
            }
        }
    }

    Ui.PanelSeparator { foreground: root.foreground }

    Ui.Button {
        text: root.tr("storage.clear")
        bordered: true
        foreground: root.foreground
        fontFamily: root.face
        onClicked: root.sheetOpen = !root.sheetOpen
    }

    Rectangle {
        id: sheet

        visible: root.sheetOpen
        width: root.width
        height: sheetColumn.height + 2 * root.sheetPad
        color: "transparent"
        radius: Style.cornerRadius
        border.width: 1
        border.color: Qt.rgba(root.foreground.r, root.foreground.g, root.foreground.b, 0.3)

        Column {
            id: sheetColumn

            x: root.sheetPad
            y: root.sheetPad
            width: sheet.width - 2 * root.sheetPad
            spacing: Style.space(10)

            Text {
                width: parent.width
                textFormat: Text.PlainText
                text: root.tr("clear.title")
                color: root.foreground
                font.bold: true
                font.family: root.face
                font.pixelSize: Style.font.subtitle
            }

            Row {
                width: parent.width
                spacing: Style.space(10)

                Ui.ToggleSwitch {
                    id: promptsSwitch
                    checked: root.clearPrompts
                    busy: root.busy
                    foreground: root.foreground
                    accent: root.accent
                    onToggled: root.clearPrompts = !root.clearPrompts
                }
                Column {
                    width: parent.width - promptsSwitch.width - parent.spacing
                    spacing: Style.space(2)

                    Text {
                        width: parent.width
                        elide: Text.ElideRight
                        textFormat: Text.PlainText
                        text: root.tr("clear.prompts")
                        color: root.foreground
                        font.bold: true
                        font.family: root.face
                        font.pixelSize: Style.font.bodySmall
                    }
                    Text {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                        text: root.tr("clear.promptsNote")
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                }
            }

            Row {
                width: parent.width
                spacing: Style.space(10)

                Ui.ToggleSwitch {
                    id: conversationsSwitch
                    checked: root.clearConversations
                    busy: root.busy
                    foreground: root.foreground
                    accent: root.accent
                    onToggled: root.clearConversations = !root.clearConversations
                }
                Column {
                    width: parent.width - conversationsSwitch.width - parent.spacing
                    spacing: Style.space(2)

                    Text {
                        width: parent.width
                        elide: Text.ElideRight
                        textFormat: Text.PlainText
                        text: root.tr("clear.conversations")
                        color: root.foreground
                        font.bold: true
                        font.family: root.face
                        font.pixelSize: Style.font.bodySmall
                    }
                    Text {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                        text: root.tr("clear.conversationsNote")
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                }
            }

            Row {
                visible: root.hasHarness
                width: parent.width
                spacing: Style.space(10)

                Ui.ToggleSwitch {
                    id: harnessesSwitch
                    checked: root.clearHarnesses
                    busy: root.busy
                    foreground: root.foreground
                    accent: root.accent
                    onToggled: root.clearHarnesses = !root.clearHarnesses
                }
                Column {
                    width: parent.width - harnessesSwitch.width - parent.spacing
                    spacing: Style.space(2)

                    Text {
                        width: parent.width
                        elide: Text.ElideRight
                        textFormat: Text.PlainText
                        text: root.tr("clear.harnesses")
                        color: root.foreground
                        font.bold: true
                        font.family: root.face
                        font.pixelSize: Style.font.bodySmall
                    }
                    Text {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                        text: root.tr("clear.harnessesNote")
                        color: root.foreground
                        opacity: 0.6
                        font.family: root.face
                        font.pixelSize: Style.font.caption
                    }
                }
            }

            Row {
                width: parent.width
                spacing: root.colGap

                Ui.Button {
                    id: sheetCancel
                    text: root.tr("clear.cancel")
                    bordered: true
                    foreground: root.foreground
                    fontFamily: root.face
                    onClicked: root.closeSheet()
                }
                Ui.Button {
                    id: sheetConfirm
                    text: root.tr("clear.confirm")
                    bordered: true
                    foreground: root.foreground
                    fontFamily: root.face
                    enabled: root.canConfirm
                    onClicked: {
                        root.clearRequested(root.clearPrompts, root.clearConversations, root.clearHarnesses && root.hasHarness);
                        root.sheetOpen = false;
                    }
                }
                Text {
                    visible: !!root.vm && root.vm.active > 0
                    width: Math.max(0, parent.width - sheetCancel.width - sheetConfirm.width - 2 * parent.spacing)
                    wrapMode: Text.WordWrap
                    textFormat: Text.PlainText
                    text: root.tr("clear.blocked")
                    color: root.foreground
                    opacity: 0.6
                    font.family: root.face
                    font.pixelSize: Style.font.caption
                }
            }
        }
    }
}
