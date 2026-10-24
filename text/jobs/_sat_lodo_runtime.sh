#!/usr/bin/env bash
# Chemins et installation automatique depuis la copie versionnee de wtpsplit.

SAT_LODO_WTPSPLIT_REV="3ef928c8e163d11dc6f6336ad610bff2ffe0bda4"

sat_lodo_set_paths() {
  local safer_root storage_root candidate
  safer_root="$(cd "${SAT_LODO_TEXT_ROOT}/.." && pwd)"
  storage_root="${SCRATCH:-$HOME}"
  SAT_LODO_RUNTIME_ROOT="${SAT_LODO_RUNTIME_ROOT:-${storage_root}/safer_sat_lodo/runtime}"
  SAT_LODO_RUN_ROOT="${SAT_LODO_RUN_ROOT:-${storage_root}/safer_sat_lodo/runs/sat12l_sm_lodo_v3}"
  SAT_WTPSPLIT_REPO="${SAT_WTPSPLIT_REPO:-${SAT_LODO_TEXT_ROOT}/vendor/wtpsplit}"
  SAT_LODO_VENV="${SAT_LODO_VENV:-${SAT_LODO_RUNTIME_ROOT}/venv}"
  SAT_LODO_REQUIREMENTS="${SAT_LODO_TEXT_ROOT}/annotation/requirements_sat_lodo.txt"
  candidate="${storage_root}/safer_private/sat12l_sm_lodo_v3"
  if [[ -z "${SAT_LODO_BUNDLE:-}" ]]; then
    if [[ -f "${candidate}/manifest.json" ]]; then
      SAT_LODO_BUNDLE="${candidate}"
    elif [[ -f "${safer_root}/annotation/lodo_sat_transfer/sat12l_sm_lodo_v3/manifest.json" ]]; then
      SAT_LODO_BUNDLE="${safer_root}/annotation/lodo_sat_transfer/sat12l_sm_lodo_v3"
    else
      SAT_LODO_BUNDLE="${safer_root}/annotation/lodo_sat_transfer/sat12l_sm_lodo_v3"
    fi
  fi
  export SAT_LODO_RUNTIME_ROOT SAT_LODO_RUN_ROOT SAT_WTPSPLIT_REPO SAT_LODO_VENV SAT_LODO_BUNDLE
  [[ -f "${SAT_LODO_BUNDLE}/manifest.json" && -f "${SAT_LODO_BUNDLE}/complete.json" ]] || {
    echo "Dossier prive introuvable ou incomplet: ${SAT_LODO_BUNDLE}" >&2
    echo "Copier sat12l_sm_lodo_v3 dans \$SCRATCH/safer_private/ ou SAFER/annotation/lodo_sat_transfer/." >&2
    return 1
  }
  [[ -f "${SAT_LODO_REQUIREMENTS}" ]] || {
    echo "Fichier de dependances introuvable: ${SAT_LODO_REQUIREMENTS}" >&2
    return 1
  }
  [[ -f "${SAT_WTPSPLIT_REPO}/SAFER_VENDOR.json" ]] || {
    echo "Copie versionnee de wtpsplit introuvable: ${SAT_WTPSPLIT_REPO}" >&2
    return 1
  }
}

sat_lodo_runtime_stamp() {
  local digest vendor_digest
  digest="$(sha256sum "${SAT_LODO_REQUIREMENTS}")"
  vendor_digest="$(sha256sum "${SAT_WTPSPLIT_REPO}/SAFER_VENDOR.json")"
  printf '%s %s %s\n' "${SAT_LODO_WTPSPLIT_REV}" "${digest%% *}" "${vendor_digest%% *}"
}

sat_lodo_runtime_packages_ok() {
  # A ready marker alone is insufficient: editable installs can disappear if
  # the environment was copied, repaired, or partially reinstalled.
  env -u PYTHONPATH "${SAT_LODO_VENV}/bin/python" - "${SAT_WTPSPLIT_REPO}" <<'PY'
import importlib.metadata
import importlib.util
import pathlib
import re
import sys

expected = (pathlib.Path(sys.argv[1]) / "wtpsplit").resolve()
requirements = expected.parents[2] / "annotation" / "requirements_sat_lodo.txt"
packages = {"torch", "wtpsplit", "adapters"}
for line in requirements.read_text(encoding="utf-8").splitlines():
    item = line.split("#", 1)[0].strip()
    if item and not item.startswith("-"):
        packages.add(re.split(r"[<>=!~;\[\s]", item, 1)[0])
for package in sorted(packages):
    importlib.metadata.version(package)
spec = importlib.util.find_spec("wtpsplit")
if spec is None or spec.origin is None or pathlib.Path(spec.origin).resolve().parent != expected:
    raise RuntimeError(f"wtpsplit is not installed from the vendored checkout: {expected}")
PY
}

sat_lodo_check_runtime() {
  local stamp_file="${SAT_LODO_RUNTIME_ROOT}/ready.txt"
  [[ -f "${stamp_file}" && -x "${SAT_LODO_VENV}/bin/python" ]] || {
    echo "Environnement absent: lancer d'abord le job smoke avec sbatch jobs/sat_lodo_smoke.sh" >&2
    return 1
  }
  [[ "$(cat "${stamp_file}")" == "$(sat_lodo_runtime_stamp)" ]] || {
    echo "L'environnement ne correspond pas au code/requirements de cette experience." >&2
    return 1
  }
  [[ -f "${SAT_WTPSPLIT_REPO}/SAFER_VENDOR.json" ]] || return 1
  sat_lodo_runtime_packages_ok || {
    echo "Environnement LoRA incomplet: relancer le precontrole pour reparer ${SAT_LODO_VENV}." >&2
    return 1
  }
}

sat_lodo_bootstrap() {
  local stamp_file="${SAT_LODO_RUNTIME_ROOT}/ready.txt"
  mkdir -p "${SAT_LODO_RUNTIME_ROOT}"
  [[ -f "${SAT_WTPSPLIT_REPO}/SAFER_VENDOR.json" ]] || return 1
  if [[ -f "${stamp_file}" && -x "${SAT_LODO_VENV}/bin/python" &&
        "$(cat "${stamp_file}")" == "$(sat_lodo_runtime_stamp)" ]]; then
    if sat_lodo_runtime_packages_ok >/dev/null 2>&1; then
      echo "Environnement LoRA deja installe: ${SAT_LODO_VENV}"
      return 0
    fi
    echo "Environnement LoRA incomplet; reinstallation du checkout wtpsplit."
    "${SAT_LODO_VENV}/bin/python" -m pip install --no-deps -e "${SAT_WTPSPLIT_REPO}"
    if sat_lodo_runtime_packages_ok >/dev/null 2>&1; then
      echo "Environnement LoRA repare: ${SAT_LODO_VENV}"
      return 0
    fi
  fi
  rm -f "${stamp_file}"
  if [[ ! -x "${SAT_LODO_VENV}/bin/python" ]]; then
    python -m venv "${SAT_LODO_VENV}"
  fi
  "${SAT_LODO_VENV}/bin/python" -m pip install --upgrade pip wheel
  "${SAT_LODO_VENV}/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  "${SAT_LODO_VENV}/bin/python" -m pip install -r "${SAT_LODO_REQUIREMENTS}"
  "${SAT_LODO_VENV}/bin/python" -m pip install --no-deps -e "${SAT_WTPSPLIT_REPO}"
  "${SAT_LODO_VENV}/bin/python" -m pip install --no-deps adapters==0.2.1
  sat_lodo_runtime_packages_ok
  "${SAT_LODO_VENV}/bin/python" -m pip freeze > "${SAT_LODO_RUNTIME_ROOT}/pip_freeze.txt"
  sat_lodo_runtime_stamp > "${stamp_file}"
}
