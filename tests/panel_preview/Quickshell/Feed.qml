pragma Singleton
import QtQuick
// Preview feed: what the stand-in Process "prints" for each local-ai subcommand, taken from one
// scenario file (the second-to-last argument). Nothing is executed and no real state is read.
//   status        object printed by `status --json` ("__NOW__" strings become the current second)
//   status_raw    text printed instead of a status object
//   status_exit   exit code of the status command (default 0)
//   status_missing  true: the status command never starts (no output, no exit code)
//   action_exit   exit code of every action (load, unload, mode, ...; default 0)
//   share         object printed by `share status --json` (absent: the command has no share)
QtObject {
  readonly property var args: Qt.application.arguments
  readonly property string scenarioFile: args.length >= 2 ? args[args.length - 2] : ""
  readonly property string fixtures: Qt.resolvedUrl("../../fixtures/panel/")
  readonly property var scenario: JSON.parse(read("file://" + scenarioFile))
  property var calls: []
  property real cardHeight: 0

  function read(url) { var x = new XMLHttpRequest(); x.open("GET", url, false); x.send(); return x.responseText }
  function now(text) { return text.split('"__NOW__"').join(String(Math.floor(Date.now() / 1000))) }
  // Fixture times are replaced by recent ones so the rows read as they would live.
  function recent(text) {
    var value = JSON.parse(text), gaps = [20, 130, 860, 2400, 7300]
    for (var i = 0; i < value.rows.length; i++)
      if (value.rows[i] && typeof value.rows[i] === "object") value.rows[i].t = Date.now() / 1000 - gaps[i % gaps.length]
    return JSON.stringify(value)
  }
  function reply(command) {
    var s = scenario
    var name = command.length > 1 ? String(command[1]) : ""
    calls = calls.concat([command.slice(1).join(" ")])
    if (name === "status") {
      if (s.status_missing === true) return { started: false, text: "", code: -1 }
      return { started: true, code: s.status_exit || 0,
               text: typeof s.status_raw === "string" ? s.status_raw : now(JSON.stringify(s.status || {})) }
    }
    if (name === "ledger") return { started: true, code: 0, text: recent(read(fixtures + "ledger_cache.json")) }
    if (name === "storage") return { started: true, code: 0, text: read(fixtures + "storage.json") }
    if (name === "cache" && command[2] === "status") return { started: true, code: 0, text: read(fixtures + "cache.json") }
    if (name === "share" && command[2] === "status")  // absent = a control command without share
      return { started: true, code: s.share ? 0 : 2, text: s.share ? JSON.stringify(s.share) : "" }
    return { started: true, code: s.action_exit || 0, text: "" }
  }
}
