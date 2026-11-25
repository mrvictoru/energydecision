# System logs

Telemetry logs from long GPU runs. Most contents are ignored by Git.

## Start here

- [Repository guide](../REPOSITORY_GUIDE.md)
- [Development guide](../docs/development.md)
- [Long-run wrapper script](../scripts/run_full_learning_baseline.sh)

`scripts/run_full_learning_baseline.sh <TAG>` writes `nvidia_smi`, `dmon`, `vmstat`, `iostat` and kernel logs under `system_logs/<TAG>/`. It writes `SAFE_TO_SHUTDOWN.txt` on completion and `CRASH_DETECTED.txt` on Xid79. Shut down only after `SAFE_TO_SHUTDOWN.txt` exists.
