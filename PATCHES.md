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

## Non-patches (evaluated and rejected)

### reading_order.rs — BERT two-column scramble
File: `crates/xberg/src/extractors/pdf/reading_order.rs`
Investigated as a fork-patch candidate during rsrch Stage 1. BERT's
section 1 first paragraph is scrambled on the markdown layout path
(sentences emitted B, A, C instead of A, B, C).

Root cause established by trace, not inference. Running
`RUST_LOG='xberg::pdf::reading_order=trace'` on BERT shows the docling
port executes (269 trace lines) with correct behavior. Every region
produces `group_count=1` — one segment group per RT-DETR hint. The
graph ordering emits roots by geometry, correctly. The scramble
originates upstream: RT-DETR emits hints that split a single
two-column paragraph across multiple regions, and geometry-based
ordering separates the sentences.

Decision: no patch. The reorder is functioning as designed; a patch
would not fix BERT and risks regressing Attention (which is currently
correct). Attention and ViT corpora are unaffected. BERT corpus
authoring remains deferred pending upstream RT-DETR improvements.

### RT-DETR bundling
Considered shipping `rtdetr/model.onnx` (161 MB) in the AAR for
offline-first operation. Rejected: APK cost not justified for a
single-device project; first-load download accepted, cache lands on
n1p2 via the `~/.cache/huggingface` bind mount. Original vendoring
patch remains available if offline-first becomes a hard requirement.
