# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Reference implementation of agent.py, all four parts filled in."""
import json

import anthropic
from anthropic.lib import files_from_dir

import state
from provided import (
    COMMANDER_MODEL, COMMANDER_SYSTEM, DATA, INVESTIGATOR_MODEL, INVESTIGATOR_SYSTEM,
    MANAGED_AGENTS_BETA, MEMORY_SEED, SKILLS, TOOLS,
    deploys, diffs, incidents, metrics, wanted_output,
)

client = anthropic.Anthropic()


# ══ PART 1 · one agent ════════════════════════════════════════════════════

# ── 1. Environment ────────────────────────────────────────────────────────
def setup_environment() -> str:
    config = {"type": "cloud", "networking": {"type": "unrestricted"}}
    return state.sync(
        "environment", config,
        create=lambda: client.beta.environments.create(
            name=state.name("env"), config=config, metadata=state.owner()),
        update=lambda env_id: client.beta.environments.update(env_id, config=config),
    )


# ── 2. Upload the log ─────────────────────────────────────────────────────
def upload_log() -> str:
    path = DATA / "app.log"

    def upload():
        with open(path, "rb") as f:
            return client.beta.files.upload(file=f)

    return state.sync("log", {"bytes": path.stat().st_size}, create=upload)


# ── 3. Skills ─────────────────────────────────────────────────────────────
def setup_skill(folder: str) -> dict:
    path = SKILLS / folder
    skill_id = state.sync(
        f"skill:{folder}", state.dir_fingerprint(path),
        # display_title must be unique in the workspace, so it carries your handle.
        create=lambda: client.beta.skills.create(
            display_title=f"{folder} ({state.handle()})", files=files_from_dir(path)),
        # Edited the folder? Publish a new version of the same skill.
        update=lambda skill_id: client.beta.skills.versions.create(skill_id, files=files_from_dir(path)),
    )
    return {"type": "custom", "skill_id": skill_id, "version": "latest"}


def _save_agent(key: str, config: dict) -> str:
    """Create the agent once; after that, a changed config becomes a new version."""
    return state.sync(
        key, config,
        create=lambda: client.beta.agents.create(**config),
        update=lambda agent_id: client.beta.agents.update(
            agent_id, version=client.beta.agents.retrieve(agent_id).version, **config),
    )


# ── 4. Agent ──────────────────────────────────────────────────────────────
def setup_agent() -> str:
    investigator_id = setup_investigator()
    return _save_agent("agent", dict(
        name=state.name("commander"),
        model=COMMANDER_MODEL,
        system=COMMANDER_SYSTEM,
        tools=TOOLS,
        skills=[setup_skill("incident-triage-runbook"), setup_skill("postmortem-report")],
        # Part 2: the roster. Pinning the version means a changed investigator
        # changes this config too, so the commander is re-saved and picks it up.
        multiagent={
            "type": "coordinator",
            "agents": [{"type": "agent", "id": investigator_id, "version": state.version("investigator")}],
        },
        metadata=state.owner(),
    ))


# ── 5. Session ────────────────────────────────────────────────────────────
def start_session(agent_id: str, env_id: str, log_file_id: str) -> str:
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        resources=[
            {"type": "file", "file_id": log_file_id, "mount_path": "app.log"},
            *setup_memory(),  # Part 3
        ],
        metadata=state.owner(),
    )
    return session.id


# ── 6. Stream loop ────────────────────────────────────────────────────────
def stream_reply(session_id: str, user_text: str):
    # Open the stream before sending, so no event from this turn is missed.
    with client.beta.sessions.events.stream(session_id) as stream:
        client.beta.sessions.events.send(
            session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": user_text}]}],
        )
        for ev in stream:
            yield ev
            if ev.type == "agent.custom_tool_use":
                # A sub-agent's call is cross-posted here with session_thread_id
                # set. Answer it the same way: the server routes by the id.
                client.beta.sessions.events.send(
                    session_id,
                    events=[{
                        "type": "user.custom_tool_result",
                        "custom_tool_use_id": ev.id,
                        "content": [{"type": "text", "text": handle_tool(ev.name, ev.input)}],
                    }],
                )
            elif ev.type == "session.status_terminated":
                return
            elif ev.type == "session.status_idle" and ev.stop_reason.type != "requires_action":
                return  # end_turn or retries_exhausted; requires_action means "waiting on you"


# ── 7. Local tool handlers ────────────────────────────────────────────────
def handle_tool(name: str, args: dict) -> str:
    if name == "list_incidents":
        return json.dumps(incidents)
    if name == "get_metrics":
        series = metrics.get(args.get("service", ""), {}).get(args.get("metric", ""))
        if series is None:
            return "unknown service or metric. valid: " + json.dumps({s: list(m) for s, m in metrics.items()})
        return json.dumps(series)
    if name == "get_recent_deploys":
        return deploys
    if name == "get_diff":
        return diffs.get(str(args.get("commit", ""))[:7], "no diff on file for that commit")
    return f"unknown tool {name}"


# ── 8. Delete session ─────────────────────────────────────────────────────
def delete_session(session_id: str) -> None:
    client.beta.sessions.delete(session_id)


# ══ PART 2 · multiagent ═══════════════════════════════════════════════════
def setup_investigator() -> str:
    return _save_agent("investigator", dict(
        name=state.name("investigator"),
        model=INVESTIGATOR_MODEL,
        system=INVESTIGATOR_SYSTEM,
        tools=TOOLS,  # a sub-agent only has the tools in its own config
        skills=[setup_skill("incident-triage-runbook")],
        metadata=state.owner(),
    ))


# ══ PART 3 · memory stores ════════════════════════════════════════════════
def setup_memory() -> list[dict]:
    def create_team_history():
        store = client.beta.memory_stores.create(
            name=state.name("team-history"),
            description="What the SRE team already knows: service map, past incidents, "
                        "known noise, postmortem conventions. Curated by humans.",
            metadata=state.owner(),
        )
        for file in sorted(MEMORY_SEED.rglob("*.md")):
            client.beta.memory_stores.memories.create(
                store.id, path="/" + file.relative_to(MEMORY_SEED).as_posix(), content=file.read_text())
        return store

    # Edited memory-seed/? The fingerprint changes and a fresh store is seeded.
    team_history = state.sync("memory:team-history", state.dir_fingerprint(MEMORY_SEED), create=create_team_history)
    notebook = state.sync("memory:notebook", "v1", create=lambda: client.beta.memory_stores.create(
        name=state.name("notebook"),
        description="The SRE agent's own notes from incidents it has worked. One file per incident.",
        metadata=state.owner(),
    ))
    return [
        {"type": "memory_store", "memory_store_id": team_history, "access": "read_only",
         "instructions": "Team knowledge. Read service-map.md and known-noise.md before you investigate, "
                         "check incidents/ for anything that looks like this one, and follow "
                         "postmortem-conventions.md in anything you write up."},
        {"type": "memory_store", "memory_store_id": notebook, "access": "read_write",
         "instructions": "Your own notebook. Check it first for incidents you have already worked. "
                         "When an investigation concludes, save a short note at "
                         "/incidents/<date>-<INCIDENT_ID>.md: root cause, how you proved it, the fix."},
    ]


# ══ PART 4 · file output ══════════════════════════════════════════════════
def download_outputs(session_id: str) -> list[tuple[str, bytes]]:
    # Anything the agent writes under /mnt/session/outputs/ is captured by the
    # Files API, scoped to the session.
    files = client.beta.files.list(scope_id=session_id, betas=[MANAGED_AGENTS_BETA])
    return [(f.filename, client.beta.files.download(f.id).read()) for f in files if wanted_output(f)]
