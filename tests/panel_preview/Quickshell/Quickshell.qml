pragma Singleton
import QtQuick
// Preview stand-in for the Quickshell singleton: no real environment is read.
QtObject {
  function env(name) { return name === "HOME" ? "/nonexistent-preview-home" : "" }
}
