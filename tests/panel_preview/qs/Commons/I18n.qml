pragma Singleton
import QtQuick
QtObject {
  readonly property bool rightToLeft: false
  function formatNumber(value, decimals) { return Number(value).toFixed(decimals === undefined ? 0 : decimals) }
}
