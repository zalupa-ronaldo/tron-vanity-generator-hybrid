# Agent workflow

This directory contains generated continuation notes for the Agent Engineering
Standard. The active policy is `.agent-policy.yaml`; the discovery artifacts are
`.agent-policy/scan.json` and `.agent-policy/module-map.json`.

Before a task, read the policy, the nearest module instructions, and the current
handoff. Run `agent-policy context MODULE` before loading a large subsystem.
After a task, run `agent-policy check`, create commit-bound `agent-policy
evidence`, and update `agent-policy handoff`. A result is `BLOCKED` when a
required command, permission, secret, or invariant is unavailable.
