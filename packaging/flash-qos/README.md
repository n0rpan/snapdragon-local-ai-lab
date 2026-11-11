# flash-qos: CPU 8 shallow idle while the Flash engine is loaded

While the Flash engine is loaded, CPU 8 should use a shallow idle state.
The knob `/sys/devices/system/cpu/cpu8/power/pm_qos_resume_latency_us` holds
100 while the engine lives and returns to 0 on unload, crash, kill or helper failure.

Files:

- `local-ai-flash-qos`: root-run helper (`hold <pid>`, `reset`, `status`). Installed as
  `/usr/local/libexec/local-ai-flash-qos`, root-owned. The install step rewrites
  `ALLOWED_UID` to the owner's uid. `hold` only accepts a process of that uid named `oxe-server`.
- `local-ai-flash-qos@.service`: system template unit. The instance name is the engine pid.
  `ExecStopPost` runs `reset`, so the knob is cleared even if the helper dies.
- `49-local-ai-flash-qos.rules`: polkit rule that lets the user in `ALLOWED_USER` start and
  stop only `local-ai-flash-qos@<pid>.service` units.
- `README.md`: this file.

Ownership: the helper is the only writer of the knob. The unprivileged controller starts
`local-ai-flash-qos@<ENGINE_PID>.service` after the engine loads and stops it on unload.

Tests: `tests/test_flash_qos_helper.py` covers the helper (as a non-root user, against a fake
sysfs and procfs) and the static content of the unit and the rule. The unit and the polkit rule
are not exercised by those tests and must be checked on the device.
