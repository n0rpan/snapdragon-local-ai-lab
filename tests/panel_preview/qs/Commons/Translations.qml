import QtQuick
// Preview stand-in: the English catalog, handed in as an object or read from `directory`.
QtObject {
  id: root
  property url directory
  property var catalog: ({})
  function load() {
    if (String(root.directory) === "" || Object.keys(root.catalog).length > 0) return
    var x = new XMLHttpRequest(); x.open("GET", String(root.directory) + "/en.json", false); x.send()
    root.catalog = JSON.parse(x.responseText)
  }
  onDirectoryChanged: load()
  Component.onCompleted: load()
  function t(key) { var v = catalog[key]; return typeof v === "string" ? v : key }
  function tn(key, n) { var v = catalog[key]; return v && typeof v === "object" ? (n === 1 ? v.one : v.other) : key }
}
