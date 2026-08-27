#!/usr/bin/env bash
set -euo pipefail

if [[ "${LAUNCHER_DRY_RUN:-false}" == true ]]; then
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WEBSHOP_SHARED_ROOT="${WEBSHOP_SHARED_ROOT:-/data/zhangdw12/work/verl-agent/agent_system/environments/env_package/webshop/webshop}"
WEBSHOP_LOCAL_ROOT="${WEBSHOP_LOCAL_ROOT:-${REPO_ROOT}/agent_system/environments/env_package/webshop/webshop}"
LOCK_FILE="${WEBSHOP_LOCAL_ROOT}.bootstrap.lock"
LOCK_DIR="${WEBSHOP_LOCAL_ROOT}.bootstrap.lock.d"
LOCK_KIND=""
CREATED_PATHS=()
CREATED_SOURCES=()
ASSETS=(
  data/items_shuffle_1000.json
  data/items_ins_v2_1000.json
  data/items_human_ins.json
  search_engine/indexes
)

cleanup() {
  local status=$?
  local index path source

  trap - EXIT INT TERM
  if (( status != 0 )); then
    for (( index=${#CREATED_PATHS[@]} - 1; index >= 0; index-- )); do
      path="${CREATED_PATHS[index]}"
      source="${CREATED_SOURCES[index]}"
      if [[ -L "${path}" ]] && [[ "$(readlink "${path}")" == "${source}" ]]; then
        rm -f "${path}"
      fi
    done
  fi
  if [[ "${LOCK_KIND}" == "mkdir" ]]; then
    rm -f "${LOCK_DIR}/pid"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

acquire_lock() {
  local attempt

  mkdir -p "$(dirname "${LOCK_DIR}")"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${LOCK_FILE}"
    if ! flock -w 60 9; then
      printf 'Timed out waiting for WebShop bootstrap lock: %s\n' "${LOCK_FILE}" >&2
      return 1
    fi
    LOCK_KIND="flock"
    return 0
  fi

  for (( attempt=0; attempt < 600; attempt++ )); do
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
      printf '%s\n' "$$" > "${LOCK_DIR}/pid"
      LOCK_KIND="mkdir"
      return 0
    fi
    sleep 0.1
  done
  printf 'Timed out waiting for WebShop bootstrap lock: %s\n' "${LOCK_DIR}" >&2
  printf 'Remove that directory only after confirming its recorded PID is no longer running.\n' >&2
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
  local_path="${WEBSHOP_LOCAL_ROOT}/${relative_path}"
  shared_path="${WEBSHOP_SHARED_ROOT}/${relative_path}"
  if [[ -L "${local_path}" && ! -e "${local_path}" ]]; then
    printf 'Broken local WebShop symbolic link: %s\n' "${local_path}" >&2
    exit 1
  fi
  if [[ -e "${local_path}" ]]; then
    validate_resource "${relative_path}" "${local_path}" "local"
  else
    if ! validate_resource "${relative_path}" "${shared_path}" "shared"; then
      printf 'Run the shared WebShop setup first: cd "%s" && ./setup.sh -d all\n' "${WEBSHOP_SHARED_ROOT}" >&2
      exit 1
    fi
  fi
done

for relative_path in "${ASSETS[@]}"; do
  local_path="${WEBSHOP_LOCAL_ROOT}/${relative_path}"
  shared_path="${WEBSHOP_SHARED_ROOT}/${relative_path}"
  if [[ -e "${local_path}" || -L "${local_path}" ]]; then
    continue
  fi
  mkdir -p "$(dirname "${local_path}")"
  ln -s "${shared_path}" "${local_path}"
  CREATED_PATHS+=("${local_path}")
  CREATED_SOURCES+=("${shared_path}")
done

for relative_path in "${ASSETS[@]}"; do
  validate_resource \
    "${relative_path}" \
    "${WEBSHOP_LOCAL_ROOT}/${relative_path}" \
    "local"
done
