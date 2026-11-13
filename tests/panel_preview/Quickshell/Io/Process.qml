import QtQuick
import Quickshell
// Preview stand-in: starts nothing. One event-loop turn after `running` is set it hands the
// scenario's text to `stdout` and reports the scenario's exit code, as the real one does.
QtObject {
  id: proc
  property var command: []
  property bool running: false
  property QtObject stdout: null
  signal exited(int exitCode, int exitStatus)
  property Timer timer: Timer { interval: 1; onTriggered: proc.finish() }
  onRunningChanged: if (running) timer.restart()
  function finish() {
    var reply = Feed.reply(proc.command)
    if (reply.started === false) { proc.running = false; return }
    if (proc.stdout) proc.stdout.finish(reply.text)
    proc.exited(reply.code, 0)
    proc.running = false
  }
}
