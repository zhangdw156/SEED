#!/usr/bin/env bash
set -euo pipefail

if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]; then
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WEBSHOP_LOCAL_ROOT="${WEBSHOP_LOCAL_ROOT:-${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop}"
WEBSHOP_SHARED_ROOT="${WEBSHOP_SHARED_ROOT:-${REPO_ROOT}/../verl-agent/agent_system/environments/env_package/webshop/webshop}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCK_DIR="${WEBSHOP_LOCAL_ROOT}.bootstrap.lock"
LOCK_HELD=0
CREATED_PATHS=()
REPLACED_PATHS=()
REPLACED_TARGETS=()
ASSETS=(
  data/items_shuffle_1000.json
  data/items_ins_v2_1000.json
  data/items_human_ins.json
  search_engine/indexes
)

relative_target() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import os
import sys

print(os.path.relpath(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])))
PY
}

canonical_path() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

cleanup() {
  local status=$?
  local index path

  trap - EXIT INT TERM
  if (( status != 0 )); then
    for (( index=${#CREATED_PATHS[@]} - 1; index >= 0; index-- )); do
      path="${CREATED_PATHS[index]}"
      [[ ! -L "${path}" ]] || rm -f "${path}"
    done
    for (( index=${#REPLACED_PATHS[@]} - 1; index >= 0; index-- )); do
      path="${REPLACED_PATHS[index]}"
      rm -f "${path}"
      ln -s "${REPLACED_TARGETS[index]}" "${path}"
    done
  fi
  if (( LOCK_HELD )); then
    rm -f "${LOCK_DIR}/pid"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

acquire_lock() {
  local attempt owner_pid

  mkdir -p "$(dirname "${LOCK_DIR}")"
  for (( attempt=0; attempt < 600; attempt++ )); do
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      printf '%s\n' "$$" > "${LOCK_DIR}/pid"
      LOCK_HELD=1
      return 0
    fi
    if [[ -r "${LOCK_DIR}/pid" ]]; then
      owner_pid="$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)"
      if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${owner_pid}" 2>/dev/null; then
        rm -f "${LOCK_DIR}/pid"
        rmdir "${LOCK_DIR}" 2>/dev/null || true
      fi
    fi
    sleep 0.1
  done
  printf 'Timed out waiting for WebShop bootstrap lock: %s\n' "${LOCK_DIR}" >&2
  return 1
}

validate_resource() {
  local relative_path=$1
  local path=$2
  local label=$3
  local segment

  if [[ "${relative_path}" == "search_engine/indexes" ]]; then
    if [[ ! -d "${path}" ]]; then
      printf 'Invalid %s WebShop index directory: %s\n' "${label}" "${path}" >&2
      return 1
    fi
    for segment in "${path}"/segments_*; do
      if [[ -f "${segment}" && -s "${segment}" ]]; then
        return 0
      fi
    done
    printf 'Invalid %s WebShop index (no non-empty segments_* file): %s\n' "${label}" "${path}" >&2
    return 1
  fi

  if [[ ! -f "${path}" || ! -s "${path}" ]]; then
    printf 'Invalid %s WebShop data file: %s\n' "${label}" "${path}" >&2
    return 1
  fi
}

acquire_lock

for relative_path in "${ASSETS[@]}"; do
  validate_resource \
    "${relative_path}" \
    "${WEBSHOP_SHARED_ROOT}/${relative_path}" \
    "shared"
done

for relative_path in "${ASSETS[@]}"; do
  local_path="${WEBSHOP_LOCAL_ROOT}/${relative_path}"
  shared_path="${WEBSHOP_SHARED_ROOT}/${relative_path}"
  link_target="$(relative_target "${shared_path}" "$(dirname "${local_path}")")"

  if [[ -L "${local_path}" ]]; then
    current_target="$(readlink "${local_path}")"
    if [[ "${current_target}" == "${link_target}" ]]; then
      continue
    fi
    if [[ "$(canonical_path "${local_path}")" != "$(canonical_path "${shared_path}")" ]] \
      && { [[ "${current_target}" != /* ]] || [[ -e "${local_path}" ]]; }; then
      printf 'Refusing to replace unrelated WebShop symlink: %s -> %s\n' \
        "${local_path}" "${current_target}" >&2
      exit 1
    fi
    REPLACED_PATHS+=("${local_path}")
    REPLACED_TARGETS+=("${current_target}")
    replacement="${local_path}.relative.$$"
    ln -s "${link_target}" "${replacement}"
    mv -f "${replacement}" "${local_path}"
    continue
  fi

  if [[ -e "${local_path}" ]]; then
    validate_resource "${relative_path}" "${local_path}" "local"
    continue
  fi

  mkdir -p "$(dirname "${local_path}")"
  ln -s "${link_target}" "${local_path}"
  CREATED_PATHS+=("${local_path}")
done

for relative_path in "${ASSETS[@]}"; do
  local_path="${WEBSHOP_LOCAL_ROOT}/${relative_path}"
  if [[ -L "${local_path}" && "$(readlink "${local_path}")" == /* ]]; then
    printf 'WebShop resource link must be relative: %s\n' "${local_path}" >&2
    exit 1
  fi
  validate_resource "${relative_path}" "${local_path}" "local"
done
