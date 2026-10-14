#!/usr/bin/env bash
set -euo pipefail
REPO="${GITHUB_REPOSITORY:?GitHub Actions repository required}"
TAG="nlp-assignment1-original-omniglot-checkpoints-20261015"
ROOT="${RUNNER_TEMP:-/tmp}/nlp-research-checkpoints"
SOURCE="$ROOT/source";OUT="$ROOT/release"
mkdir -p "$SOURCE" "$OUT"

wait_run() {
  local id="$1" status conclusion i
  for i in $(seq 1 140); do
    status=$(gh api "repos/$REPO/actions/runs/$id" --jq '.status')
    conclusion=$(gh api "repos/$REPO/actions/runs/$id" --jq '.conclusion // "pending"')
    printf '%s: check %s -> %s/%s\n' "$id" "$i" "$status" "$conclusion"
    if [ "$status" = completed ]; then
      if [ "$conclusion" = success ]; then return 0; fi
      echo "Required workflow $id failed with $conclusion" >&2
      return 1
    fi
    sleep 90
  done
  echo "Timed out waiting for $id" >&2
  return 1
}
wait_run 38642335173
wait_run 38648489063

# Each input is a COMPLETE original GitHub Actions artifact collection,
# not a reconstruction from numerical summaries.
while read -r label id; do
  mkdir -p "$SOURCE/$label"
  gh run download "$id" -R "$REPO" --dir "$SOURCE/$label"
  find "$SOURCE/$label" -type f -name 'model.eqx' -o -name 'extended_g0_model.eqx' | head -n 5
  if ! find "$SOURCE/$label" -type f \( -name 'model.eqx' -o -name 'extended_g0_model.eqx' \) | grep -q .; then
    echo "Missing model weights in source: $label" >&2
    exit 1
  fi
  tar -czf "$OUT/$label.tar.gz" -C "$SOURCE" "$label"
  echo "RELEASE_SOURCE=$label run=$id archive=$OUT/$label.tar.gz"
done <<'SOURCES'
original_base_G0 38612082174
original_extended_G0 38630071147
original_base_G10_4arms 38632979396
original_extended_G10_4arms 38633208767
independent_extended_G10 38641472801
independent_base_G10 38642335173
primary_extended_repairs 38641963903
independent_extended_repairs 38648489063
SOURCES

(cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS)
cat > "$OUT/README.txt" <<'NOTES'
NLP Assignment 1 original Omniglot model research checkpoint preservation.
Real GitHub Actions artifacts packaged by SOURCE RUN, including checkpoint
weights, metadata, validation histories and generation trajectories.
Scripts and verified nonbinary G10 plots/results are in the repository branch
assignment-roadmap-g100-20261015 and draft PR #4.
G3 final test was opened once. G10 results, independent replications and
all new repair interventions are validation only.
Two different training exposure budgets: original Base 1M/generation,
Extended 300k/generation. The low-budget G100 curiosity is not part of
this scientifically comparable full-budget checkpoint snapshot.
Original training data is from the pinned upstream source submodule.
This release is PUBLIC because the repository is public.
Do not conflate target-only recursion with synthetic input-prefix generation
or constant-size 50/50 mixing with historical accumulation.
NOTES
if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "$OUT"/*.tar.gz "$OUT/SHA256SUMS" "$OUT/README.txt" -R "$REPO" --clobber
else
  gh release create "$TAG" "$OUT"/*.tar.gz "$OUT/SHA256SUMS" "$OUT/README.txt" \
    -R "$REPO" \
    --target "${GITHUB_SHA}" \
    --title "NLP Assignment 1 — original Omniglot model checkpoints (October 2026)" \
    --notes "Research reproducibility snapshot of the actual training artifacts. Original frozen G3 final test was previously used; later G10 and repair results are validation only. Refer to docs/OVERNIGHT_EXECUTION_AND_REPO_STATE_20261015.md." \
    --latest=false
fi
gh release view "$TAG" -R "$REPO" --json url,assets --jq '{url:.url,assets:[.assets[].name]}'
