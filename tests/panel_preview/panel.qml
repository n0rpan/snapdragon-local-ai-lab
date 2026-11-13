// Offscreen preview of the whole panel (the plugin's Panel.qml) against stand-ins for Omarchy's
// kit and for Quickshell's Process. Not part of the plugin; nothing is executed or installed.
// Run: tests/panel_preview/render_panel.sh <scenario.json> <out-prefix>
// The scenario (see Quickshell/Feed.qml) may also set: view (the tab the panel starts on),
// act (arguments of one panel action to run first), mode (a context mode to request),
// share_steps (clicks on the sharing row: "pick:<mode>", "confirm", "cancel").
// For each tab it writes <out-prefix>-<tab>.png and logs one "PANEL-TEXTS {json}" line with the
// visible texts and the tooltip texts.
import QtQuick
import Quickshell
import "../../src/snapdragon_ai_lab/omarchy_plugin" as Plugin

Rectangle {
  id: root
  readonly property var args: Qt.application.arguments
  readonly property string outPrefix: args[args.length - 1]
  readonly property var scenario: Feed.scenario
  readonly property string startView: ["models", "activity", "storage"].indexOf(scenario.view) >= 0 ? scenario.view : "models"
  readonly property var order: [startView].concat(["models", "activity", "storage"].filter(function(v) { return v !== root.startView }))
  property int step: 0

  width: 488
  height: Math.max(240, Feed.cardHeight + 40)
  color: "#0e0e14"

  Plugin.Panel { id: panel; x: 20; y: 20; view: root.startView }

  function probe(item, out) {
    if (!item || item.visible !== true) return
    if (item.textFormat !== undefined && typeof item.text === "string" && item.text !== "") out.texts.push(item.text)
    if (typeof item.tooltipText === "string" && item.tooltipText !== "") out.tips.push(item.tooltipText)
    var kids = item.children
    for (var i = 0; i < kids.length; i++) {
      if (kids[i].previewToolTip === true) {
        if (typeof kids[i].text === "string" && kids[i].text !== "") out.tips.push(kids[i].text)
      } else {
        root.probe(kids[i], out)
      }
    }
  }
  // The sharing row, driven as its own buttons drive it: "pick:<mode>", "confirm", "cancel".
  function find(item, name) {
    if (!item) return null
    if (item.objectName === name) return item
    for (var i = 0; i < item.children.length; i++) {
      var hit = root.find(item.children[i], name)
      if (hit) return hit
    }
    return null
  }
  function shareSteps(steps) {
    var row = root.find(panel, "localAiShare")
    for (var i = 0; i < steps.length; i++) {
      if (steps[i] === "confirm") row.confirm()
      else if (steps[i] === "cancel") row.cancel()
      else row.pick(String(steps[i]).split(":")[1])
    }
  }
  function capture() {
    var out = { view: panel.view, phase: panel.vm.phase, notice: panel.notice, texts: [], tips: [], calls: Feed.calls }
    root.probe(panel, out)
    console.log("PANEL-TEXTS " + JSON.stringify(out))
    root.grabToImage(function(result) {
      result.saveToFile(root.outPrefix + "-" + panel.view + ".png")
      root.step += 1
      if (root.step >= root.order.length) { Qt.quit(); return }
      panel.view = root.order[root.step]
      again.restart()
    })
  }

  Component.onCompleted: panel.open()
  Timer {
    interval: 200; running: true
    onTriggered: {
      if (Array.isArray(root.scenario.act)) panel.act(root.scenario.act)
      else if (typeof root.scenario.mode === "string") panel.setMode(root.scenario.mode)
      else if (typeof root.scenario.share_to === "string") panel.setShare(root.scenario.share_to)
      else if (Array.isArray(root.scenario.share_steps)) root.shareSteps(root.scenario.share_steps)
    }
  }
  // Later than the status command's 500 ms grace, so a command that never starts has settled.
  Timer { interval: 1400; running: true; onTriggered: root.capture() }
  Timer { id: again; interval: 400; onTriggered: root.capture() }
}
