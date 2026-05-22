---
display_name: "XPolicyLab Import Policy"
short_description: "Import and adapt external policy repositories into XPolicyLab."
default_prompt: "Use this skill to import a policy repository into XPolicyLab, adapt its files to the project policy interface, and validate the integration."
---

Invoke the `xpolicylab-import-policy` skill when the user asks to:
- Clone or copy an external policy repository into `XPolicyLab/policy/`
- Scaffold or adapt policy files (`model.py`, `deploy.py`, `deploy.yml`, `install.sh`, etc.)
- Reproduce or audit an upstream policy's dependencies, observation/action format, or image preprocessing
- Validate a new policy integration via the debug-client path

Full behavior is defined in `skills/xpolicylab-import-policy/SKILL.md`.
