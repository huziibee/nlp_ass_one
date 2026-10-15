#!/usr/bin/env bash
set -euo pipefail
repo="${GITHUB_REPOSITORY:?}"
tag="nlp-assignment1-mechanistic-followups-20261015"
root="${RUNNER_TEMP:-/tmp}/followup-archive"
mkdir -p "$root/artifacts" "$root/release"
status="$root/release/EXPERIMENT_STATUS.txt"
: > "$status"
for spec in "dose_response:38655623103" "joint_heads:38655884354" "three_seed_onset:38655681721" "budget_match_base:38655728846" "growing_historical_corpus:38655888849" "factorial_single_field:38656350306"; do
  name="${spec%%:*}"; run_id="${spec##*:}"
  completed=no; conclusion=pending
  for attempt in $(seq 1 150); do
    completed=$(gh api "repos/$repo/actions/runs/$run_id" --jq .status)
    conclusion=$(gh api "repos/$repo/actions/runs/$run_id" --jq '.conclusion // "pending"')
    echo "$name $run_id: $completed/$conclusion (poll $attempt)"
    if [ "$completed" = completed ]; then break; fi
    sleep 90
  done
  if [ "$conclusion" != success ]; then
    echo "$name run=$run_id status=$completed conclusion=$conclusion, NOT a completed verified experiment" | tee -a "$status"
    continue
  fi
  mkdir -p "$root/artifacts/$name"
  gh run download "$run_id" -R "$repo" --dir "$root/artifacts/$name"
  tar -czf "$root/release/$name.tar.gz" -C "$root/artifacts" "$name"
  echo "$name run=$run_id status=completed conclusion=success source=GitHub Actions actual artifact" | tee -a "$status"
done
(
  cd "$root/release"
  sha256sum ./*.tar.gz > SHA256SUMS
)
cat > "$root/release/README.txt" <<'TXT'
Original-feature Omniglot followup experiment artifacts.
Source repository: huziibee/nlp_ass_one, followup PR #5.
Includes source-run original scientific results, and model.eqx checkpoints
when the corresponding experiment emitted them. Some runs can be exploratory
and non-full-budget; refer to each artifact's JSON metadata and
EXPERIMENT_STATUS.txt. No protected original final test is reopened.
TXT
if gh release view "$tag" -R "$repo" >/dev/null 2>&1; then
  gh release upload "$tag" "$root/release"/* -R "$repo" --clobber
else
  gh release create "$tag" "$root/release"/* -R "$repo" \
    --target "${GITHUB_SHA}" \
    --title "NLP assignment — mechanistic followup experiment checkpoints" \
    --notes "Verified source GitHub Actions experiment artifacts for partial repair, joint-head knockout, generation onset, budget-matched Base, accumulated historical source examples, and single-field factorial repair. Some may be incomplete; consult EXPERIMENT_STATUS.txt and individual metadata." \
    --latest=false
fi
gh release view "$tag" -R "$repo" --json url,assets --jq '{url:.url,names:[.assets[].name]}'
