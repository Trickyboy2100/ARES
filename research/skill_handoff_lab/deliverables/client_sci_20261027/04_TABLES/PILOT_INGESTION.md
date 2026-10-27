Actual-run interface: CSV rows only, no fabricated bars.

columns: method,seed,split,family,n,raw_successes,independent_successes,
interactions,run,source_sha256,audit_sha256. independent_successes may be
blank ONLY when audit is incomplete; that row is displayed as Not tested.

No row is extrapolated to missing seeds. Use --pilot-csv after completed runs and audits.
