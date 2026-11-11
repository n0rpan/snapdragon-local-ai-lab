# flash-qos: CPU 8 shallow idle while the Flash engine is loaded

While the Flash engine is loaded, CPU 8 should use a shallow idle state.
The knob `/sys/devices/system/cpu/cpu8/power/pm_qos_resume_latency_us` holds
100 while the engine lives and returns to 0 on unload, crash, kill or helper failure.

Files:

- `local-ai-flash-qos`: root-run helper (`hold <pid>`, `reset`, `status`). Installed as
  `/usr/local/libexec/local-ai-flash-qos`, root-owned. The public source has an unconfigured
  `ALLOWED_UID = None`; `hold` refuses it without writing the knob. The install step renders
  the validated installing account's UID. `hold` only accepts a process of that UID named `oxe-server`.
- `local-ai-flash-qos@.service`: system template unit. The instance name is the engine pid.
  `ExecStopPost` runs `reset`, so the knob is cleared even if the helper dies.
- `49-local-ai-flash-qos.rules`: polkit rule that lets the user in `ALLOWED_USER` start and
  stop only `local-ai-flash-qos@<pid>.service` units. Its unconfigured `null` source grants
  nobody access; install-root.sh renders the same account used for the helper.
- `README.md`: this file.

Ownership: the helper is the only writer of the knob. The unprivileged controller starts
`local-ai-flash-qos@<ENGINE_PID>.service` after the engine loads and stops it on unload.

The Flash stage generator writes `system/install-root.sh`. Run it through sudo as the account
that owns the local inference controller; the installer uses `SUDO_USER`. A root shell without
a non-root `SUDO_USER` must name the account explicitly:

```sh
sudo /path/to/stage/system/install-root.sh
# Or, from an administrative/root context:
/path/to/stage/system/install-root.sh --user localai
```

The requested name must resolve consistently through both `getent passwd` and `id`, with a
nonzero UID and an exact canonical account-name match. Missing, unknown, root and malformed
accounts fail before any system file is installed. The helper UID and polkit user are rendered
together into private temporary files; source templates are not modified. The installer writes
the same three root-owned files and reloads systemd metadata without starting a unit. It does
not grant group-wide access, change the unit-name/verb restrictions, or select a user by a fixed UID.

Tests: `tests/test_flash_qos_helper.py` covers the helper (as a non-root user, against a fake
sysfs and procfs) and the static content of the unit and the rule. The unit and the polkit rule
are not exercised by those tests and must be checked on the device.
