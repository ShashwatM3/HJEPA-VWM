# For coding agents

**Read [`AGENT_FILES/AGENTS.md`](AGENT_FILES/AGENTS.md) first.** It is the focused entry point for this repository: what the project is, every file and folder, mandatory read order, and how to behave.

Do not infer architecture from generic ML patterns. Follow the agent docs under `AGENT_FILES/`.

For any task involving SSH, RunPod, a remote GPU server, `tmux`, remote commands, or launching,
resuming, stopping, or monitoring training, load
[`run-remote-experiment`](.agents/skills/run-remote-experiment/SKILL.md) and read
[`AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md`](AGENT_FILES/GUIDE_AGENT_SSH_ACCESS.md) before the first
remote command.
