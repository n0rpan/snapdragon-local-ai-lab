#!/bin/sh
# Renders the whole panel offscreen from one scenario file, with stand-ins for Omarchy's kit
# (tests/panel_preview/qs) and for Quickshell's Process (tests/panel_preview/Quickshell): no
# command runs, no shell is touched. Writes <out-prefix>-{models,activity,storage}.png.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
QT_QPA_PLATFORM=offscreen QT_FORCE_STDERR_LOGGING=1 QML_XHR_ALLOW_FILE_READ=1 \
  timeout 40 qml6 -I "$here" "$here/panel.qml" -- "$(realpath "$1")" "$(realpath -m "$2")"
