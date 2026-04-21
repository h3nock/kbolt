# Schedule

## Synopsis

```bash
kbolt schedule add [OPTIONS]
kbolt schedule status
kbolt schedule remove [ID | --all | --space <SPACE> [--collection <COLLECTIONS>]]
```

## What schedule does

`schedule` runs the same indexing work as `kbolt update`, but on an automatic trigger.

The scope determines what gets updated:

- no `--space`: all spaces
- `--space work`: everything in the `work` space
- `--space work --collection api,docs`: only those collections in `work`

## `add`

Use `add` to create a new schedule.

### Trigger forms

Interval trigger:

```bash
kbolt schedule add --every 30m
kbolt schedule add --every 2h
```

Daily trigger:

```bash
kbolt schedule add --at 09:00
kbolt schedule add --at 3pm
```

Weekly trigger:

```bash
kbolt schedule add --on Mon,Fri --at 09:00
```

### Scope options

- `--space <SPACE>`: restrict the schedule to one space
- `--collection <COLLECTIONS>`: restrict the schedule to specific collections; requires `--space`

Examples:

```bash
kbolt schedule add --every 30m
kbolt schedule add --at 09:00 --space work
kbolt schedule add --on Mon,Fri --at 09:00 --space work --collection api,docs
```

### Validation rules

- `--every` accepts only minute or hour intervals such as `30m` or `2h`
- interval schedules must be at least `5m`
- daily and weekly times must parse as a valid time
- weekly schedules require at least one weekday
- `--collection` requires `--space`
- the top-level `--space` flag is rejected; pass `--space` to `schedule add` or `schedule remove` instead
- you cannot add two schedules with the same trigger and scope

## `status`

Use `status` to inspect configured schedules and their last run state:

```bash
kbolt schedule status
```

Each entry shows:

- the schedule ID, such as `s1`
- the trigger
- the scope
- the backend
- the current state
- `last_started`
- `last_finished`
- `last_result`
- `last_error` when the last run failed

State values:

- `installed`: the configured schedule is installed on the current backend
- `drifted`: the configured schedule exists in `schedules.toml`, but the backend installation no longer matches it
- `target_missing`: the schedule points at a space or collection that no longer exists

If a schedule is `drifted`, remove it by ID and add it again so `kbolt` can replace the backend entry.

Last-result values:

- `success`: the scheduled update finished successfully
- `skipped_lock`: another `kbolt` process already held the global operation lock
- `failed`: the update run failed and `last_error` contains the error message

`status` also reports `orphans`, which are backend schedule entries that no longer exist in the saved catalog.

A successful `schedule add` or `schedule remove` reconciles the managed backend state, so stale managed entries are cleaned up during the next change to the saved schedule set.

## `remove`

Use `remove` to delete schedules:

```bash
kbolt schedule remove s1
kbolt schedule remove --all
```

You can also remove by scope:

```bash
kbolt schedule remove --space work
kbolt schedule remove --space work --collection api
```

Notes:

- scope-based removal succeeds only when the scope matches exactly one saved schedule
- if multiple schedules match the scope, remove by ID instead

## Backend behavior

`kbolt` installs schedules on the native user-level scheduler for the current platform:

- macOS: `launchd`
- Linux: `systemd-user`

If the current platform does not support scheduling, schedule commands fail.

`schedule status` compares the saved catalog with the installed backend entries, which is why it can report drifted schedules and orphans.

## Related pages

- [CLI overview](../cli-overview.md)
- [Content management](content-management.md)
- [Data locations](../../operations/data-locations.md)
