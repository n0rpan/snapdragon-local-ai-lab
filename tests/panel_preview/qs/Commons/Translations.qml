import QtQuick
// Preview stand-in: English catalog handed in as an object.
QtObject {
  property url directory
  property var catalog: ({})
  function t(key) { var v = catalog[key]; return typeof v === "string" ? v : key }
  function tn(key, n) { var v = catalog[key]; return v && typeof v === "object" ? (n === 1 ? v.one : v.other) : key }
}
