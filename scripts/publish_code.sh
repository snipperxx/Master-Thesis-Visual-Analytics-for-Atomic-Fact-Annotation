#!/usr/bin/env bash
# Publish the CODE of the current commit to GitHub, and nothing else.
#
# `origin` is a public repository. The thesis, the gold, the data, the run logs,
# the guideline prompts, the docs (CHANGELOG, correspondence) and CLAUDE.md stay
# local. Local `main` keeps the full history and is never pushed; this script
# builds a code-only snapshot of HEAD on top of the public history and pushes
# that to origin/main. The local branch `public` mirrors what was pushed.
#
#   scripts/publish_code.sh            # build, show the file list, push
#   scripts/publish_code.sh --dry-run  # build and show, do not push
#   scripts/publish_code.sh --fresh    # snapshot with NO parent, into `public`
#                                      # only; for a new empty repo (prints the
#                                      # push command, never pushes itself)
#
# Never merge `public` or origin/main into local `main`: the public tree lacks
# every non-code path, so a merge would delete them from the working tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# What counts as code. Anything not listed here is never published.
CODE_PATHS=(
  src scripts ui tests cypher assets
  Dockerfile .dockerignore docker-compose.yml docker-compose.override.yml
  requirements.txt run_dev.sh stop_dev.sh
  .gitignore .env.example README.md
)
# Files under CODE_PATHS that are really data: the gold, written as Python.
EXCLUDE_PATHS=(
  scripts/gold_hires_spec.py              # the whole high-resolution gold
  scripts/preannotate_topup_segments.py   # 25 gold pre-annotations inline
)
# Content that must never appear in the published tree.
FORBIDDEN_RE='dbvis\.de|@gmail\.com|OLLAMA_CLOUD_API_KEY=[A-Za-z0-9_-]{8,}'

DRY_RUN=0
FRESH=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --fresh)   FRESH=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

src_commit=$(git rev-parse HEAD)
exclude_re="^($(IFS='|'; echo "${EXCLUDE_PATHS[*]}"))\$"
code_re="^($(printf '%s\n' "${CODE_PATHS[@]}" | sed 's/\./[.]/g' | paste -sd'|'))(/|\$)"

if (( FRESH )); then
  parent=""
else
  git fetch -q origin main
  parent=$(git rev-parse origin/main)
  # The public history must itself be code-only, or every snapshot on top of
  # it keeps the leaked files one click away. (2026-09-23: local main, thesis
  # and all, had been pushed to origin/main.)
  leaked=$(git ls-tree -r --name-only "$parent" \
    | awk -v code="$code_re" -v ex="$exclude_re" '$0 !~ code || $0 ~ ex' || true)
  if [[ -n "$leaked" ]]; then
    echo "REFUSING: origin/main carries non-code paths, e.g.:" >&2
    echo "$leaked" | head -10 >&2
    echo "($(echo "$leaked" | wc -l) paths.) Building on it would keep them public." >&2
    echo "Recreate the GitHub repo empty and publish with --fresh." >&2
    exit 1
  fi
fi

tmp_index="$(git rev-parse --git-dir)/publish_code.index"
rm -f "$tmp_index"
trap 'rm -f "$tmp_index"' EXIT
git ls-tree -r "$src_commit" -- "${CODE_PATHS[@]}" \
  | awk -F'\t' -v re="$exclude_re" '$2 !~ re' \
  | GIT_INDEX_FILE="$tmp_index" git update-index --index-info
tree=$(GIT_INDEX_FILE="$tmp_index" git write-tree)

if [[ -n "$parent" && "$tree" == "$(git rev-parse "$parent^{tree}")" ]]; then
  echo "origin/main already has this code; nothing to publish."
  exit 0
fi

if git grep -n -I -E "$FORBIDDEN_RE" "$tree" -- . >/dev/null; then
  echo "REFUSING: forbidden content in the code tree:" >&2
  git grep -n -I -E "$FORBIDDEN_RE" "$tree" -- . >&2
  exit 1
fi

# The README is the public front page: every relative link and image in it must
# resolve inside the code-only tree (it used to point at CLAUDE.md and doc/).
missing=()
while IFS= read -r target; do
  [[ -z "$target" ]] && continue
  git cat-file -e "$tree:$target" 2>/dev/null || missing+=("$target")
done < <(git show "$tree:README.md" \
  | grep -oE '\]\([^)[:space:]]+' | sed 's/^](//' \
  | grep -vE '^(https?:|mailto:|#)' | sed 's/#.*//' | sort -u || true)
if (( ${#missing[@]} )); then
  echo "REFUSING: README.md links to paths that are not published:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

n_files=$(git ls-tree -r --name-only "$tree" | wc -l)
echo "Code-only tree from $(git rev-parse --short "$src_commit"): $n_files files"
[[ -n "$parent" ]] && git diff --stat "$parent" "$tree" | tail -1

msg="Code snapshot of $(git rev-parse --short "$src_commit") ($(date -u +%F))

Code only; thesis, data, docs and prompts are not published."
if [[ -n "$parent" ]]; then
  commit=$(git commit-tree "$tree" -p "$parent" -m "$msg")
else
  commit=$(git commit-tree "$tree" -m "$msg")
fi

if (( DRY_RUN )); then
  echo "dry run: would publish $commit"
  exit 0
fi
if (( FRESH )); then
  git branch -f public "$commit"
  echo "local branch public = $(git rev-parse --short "$commit") (no parent)."
  echo "Push it to the NEW, EMPTY repo with:"
  echo "  git -c credential.helper=manager push origin public:main"
  exit 0
fi
git push origin "$commit:refs/heads/main"
git branch -f public "$commit"
echo "published $(git rev-parse --short "$commit") to origin/main (local branch: public)"
