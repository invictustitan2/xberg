# Patches vs upstream xberg

Fork of https://github.com/xberg-io/xberg for the rsrch project.
Rebase policy: pinned to upstream release tags. Each rebase re-verifies
every entry below.

## Active patches

### P1 — CI ownership guard on Android job
File: `.github/workflows/ci-mobile.yaml`
Upstream line 60 guards `android-check` with
`if: github.repository == 'xberg-io/xberg'`. On a personal fork this
evaluates false and the Android build is not CI-verified.
Patch: replace with our fork coordinate.

Applied: Stage 0, 2026-09-21
Verified: guard correct and workflow enabled on fork, but no runs
queue. Likely cause: upstream reusable actions under xberg-io/actions
are private to the org and unresolvable on the fork. Not blocking.
Re-verify if upstream's actions become public.

## Planned patches (not yet applied)

### P2 — Feature trim for rsrch
File: `crates/xberg-jni/Cargo.toml`
Trim the ~65-feature list on line 40 to the subset the rsrch cockpit
uses (PDF ingestion, layout detection, extraction-to-markdown, async).
Deferred to rsrch Stage 3.

## Rejected

### RT-DETR bundling
Considered shipping `rtdetr/model.onnx` (161 MB) in the AAR for
offline-first operation. Rejected: APK cost not justified; first-load
download is acceptable.
