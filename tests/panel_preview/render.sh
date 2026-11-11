#!/bin/sh
# Renders the view components offscreen with stand-ins for Omarchy's kit (tests/panel_preview/qs).
# Checks that the QML loads and lays out; the real kit and theme are only in the running shell.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
QT_QPA_PLATFORM=offscreen QT_FORCE_STDERR_LOGGING=1 QML_XHR_ALLOW_FILE_READ=1 \
  timeout 30 qml6 -I "$here" "$here/preview.qml" -- "$(realpath "$1")" "$(realpath -m "$2")"
