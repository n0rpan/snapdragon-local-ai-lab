import QtQuick
// Preview stand-in for the kit's bar-widget panel root: open state and settings only.
Item {
  id: root
  property QtObject bar: null
  property string moduleName: ""
  property string ipcTarget: ""
  property var settings: ({})
  property bool opened: false
  function open() { opened = true }
  function close() { opened = false }
  function toggle() { opened = !opened }
  function switchPanel(direction) { return false }
  function setting(name, fallback) {
    var value = settings ? settings[name] : undefined
    return value === undefined || value === null ? fallback : value
  }
}
