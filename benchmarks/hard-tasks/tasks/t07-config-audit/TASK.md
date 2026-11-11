The svc service in this directory is started by svc.main.run(cfg), where cfg is a svc.config.Config.

Ops needs the list of configuration keys that svc.main.run() cannot run without. Read the code and write ANSWER.json in the project root, in this form:

{"required": ["some.key", "another.key"]}

The "required" value must be a JSON list of key strings, sorted alphabetically.

Definition: a key is REQUIRED if some read of it in code reachable from svc.main.run() can raise KeyError when the key is absent, regardless of other branches. Keys read only with a default, or only in code that run() never reaches, are not required.

Do not modify the svc package.
