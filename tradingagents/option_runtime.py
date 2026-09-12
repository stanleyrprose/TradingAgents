"""Production runtime install, activation, rollback, and health helpers for option daily ops."""

from __future__ import annotations

import hashlib
import io
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tarfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_RUNTIME_ROOT = Path.home() / ".tradingagents" / "options-runtime"
_MANIFEST_NAME = "release.json"
_RUNNER_NAME = "options-daily-runner"


@dataclass(frozen=True)
class OptionRuntimeRelease:
    git_sha: str
    git_short: str
    dependency_fingerprint: str
    env_relpath: str
    created_at: str
    python_version: str


@dataclass(frozen=True)
class OptionRuntimeHealth:
    status: str
    current_release: str | None
    issues: tuple[str, ...]
    runner_path: str
    manifest_path: str | None = None


def default_runtime_root() -> Path:
    return _DEFAULT_RUNTIME_ROOT


def runtime_runner_path(runtime_root: str | os.PathLike[str] | None = None) -> Path:
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    return root / "bin" / _RUNNER_NAME


def scheduler_binding_issues(
    *,
    runtime_root: str | os.PathLike[str] | None = None,
    launch_agent_path: str | os.PathLike[str],
) -> tuple[str, ...]:
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    plist_path = Path(launch_agent_path).expanduser()
    if not plist_path.exists():
        return (f"LaunchAgent plist is missing: {plist_path}",)
    try:
        payload = plistlib.loads(plist_path.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return (f"LaunchAgent plist is invalid: {plist_path}",)
    issues: list[str] = []
    arguments = payload.get("ProgramArguments")
    expected_runner = str(runtime_runner_path(root).absolute())
    if not isinstance(arguments, list) or not arguments:
        issues.append("LaunchAgent ProgramArguments are missing")
    else:
        if arguments[0] != expected_runner:
            issues.append("LaunchAgent is not bound to the hardened runtime runner")
        if "--send" not in arguments:
            issues.append("LaunchAgent is not configured for production SEND mode")
        if "--telegram-config" not in arguments:
            issues.append("LaunchAgent does not reference the secure Telegram config file")
    environment = payload.get("EnvironmentVariables", {})
    if isinstance(environment, dict) and any(str(key).startswith("TRADINGAGENTS_TG_") for key in environment):
        issues.append("LaunchAgent must not embed Telegram credentials in environment variables")
    return tuple(issues)


def _run(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def _tracked_clean(repo_root: Path) -> None:
    result = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root)
    if result.stdout.strip():
        raise ValueError("tracked worktree changes exist; commit or restore them before runtime install")


def _git_sha(repo_root: Path) -> tuple[str, str | None]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    upstream = _run(["git", "rev-parse", "@{u}"], cwd=repo_root, check=False)
    upstream_sha = upstream.stdout.strip() if upstream.returncode == 0 else None
    return head, upstream_sha


def dependency_fingerprint(*, python_version: str, freeze_text: str) -> str:
    dependency_lines = []
    for raw_line in freeze_text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if not line:
            continue
        if lowered.startswith("tradingagents==") or "#egg=tradingagents" in lowered:
            continue
        dependency_lines.append(line)
    canonical = f"python={python_version}\n" + "\n".join(dependency_lines) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _freeze_environment(source_python: Path, repo_root: Path) -> tuple[str, str]:
    version = _run(
        [str(source_python), "-c", "import platform; print(platform.python_version())"],
        cwd=repo_root,
    ).stdout.strip()
    freeze = _run([str(source_python), "-m", "pip", "freeze", "--all"], cwd=repo_root).stdout
    return version, freeze


def _extract_git_head(repo_root: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination, filter="data")


def _site_packages(python: Path, cwd: Path) -> Path:
    result = _run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        cwd=cwd,
    )
    return Path(result.stdout.strip())


def _sanitize_copied_environment(site_packages: Path) -> None:
    for pattern in (
        "__editable__.tradingagents-*.pth",
        "__editable___tradingagents_*_finder.py",
        "__editable___tradingagents_*_finder.pyc",
    ):
        for item in site_packages.glob(pattern):
            item.unlink(missing_ok=True)
    for item in site_packages.glob("tradingagents-*.dist-info"):
        if item.is_dir():
            shutil.rmtree(item)


def _ensure_runtime_env(
    *,
    runtime_root: Path,
    repo_root: Path,
    source_python: Path,
    fingerprint: str,
) -> Path:
    env_root = runtime_root / "envs" / fingerprint
    venv = env_root / "venv"
    env_python = venv / "bin" / "python"
    if env_python.exists():
        return venv

    env_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = env_root.parent / f".{fingerprint}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        subprocess.run([str(source_python), "-m", "venv", str(temporary / "venv")], check=True)
        source_site = _site_packages(source_python, repo_root)
        dest_python = temporary / "venv" / "bin" / "python"
        dest_site = _site_packages(dest_python, repo_root)
        shutil.copytree(source_site, dest_site, dirs_exist_ok=True)
        _sanitize_copied_environment(dest_site)
        os.replace(temporary, env_root)
    finally:
        if temporary.exists():
            _remove_tree(temporary)
    return venv


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _chmod_readonly_tree(root: Path) -> None:
    for item in root.rglob("*"):
        if item.is_symlink():
            continue
        mode = 0o555 if item.is_dir() else 0o444
        os.chmod(item, mode)
    os.chmod(root, 0o555)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for item in root.rglob("*"):
        if item.is_symlink():
            continue
        with suppress(OSError):
            os.chmod(item, 0o755 if item.is_dir() else 0o644)
    with suppress(OSError):
        os.chmod(root, 0o755)
    shutil.rmtree(root)


def _symlink_atomic(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _read_manifest(release_dir: Path) -> OptionRuntimeRelease:
    path = release_dir / _MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return OptionRuntimeRelease(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid runtime manifest: {path}") from exc


def _runner_source(runtime_root: Path, base_python: Path) -> str:
    root_literal = repr(str(runtime_root))
    return f"""#!{base_python}\nfrom __future__ import annotations\n\nimport json\nimport os\nimport sys\nfrom pathlib import Path\n\nROOT = Path({root_literal})\ncurrent = (ROOT / 'current').resolve(strict=True)\nmanifest = json.loads((current / 'release.json').read_text(encoding='utf-8'))\nenv_python = ROOT / manifest['env_relpath'] / 'bin' / 'python'\napp = current / 'app'\nscript = app / 'scripts' / 'options_daily_ops.py'\nenvironment = os.environ.copy()\nenvironment['PYTHONPATH'] = str(app)\nenvironment['PYTHONDONTWRITEBYTECODE'] = '1'\nos.chdir(app)\nos.execve(str(env_python), [str(env_python), str(script), *sys.argv[1:]], environment)\n"""


def _write_runner(runtime_root: Path, base_python: Path) -> Path:
    runner = runtime_runner_path(runtime_root)
    runner.parent.mkdir(parents=True, exist_ok=True)
    temporary = runner.with_name(f".{runner.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(_runner_source(runtime_root, base_python), encoding="utf-8")
        os.chmod(temporary, 0o755)
        os.replace(temporary, runner)
    finally:
        if temporary.exists():
            temporary.unlink()
    return runner


def activate_release(runtime_root: str | os.PathLike[str], release_name: str) -> OptionRuntimeRelease:
    root = Path(runtime_root).expanduser()
    releases = root / "releases"
    matches = [p for p in releases.iterdir() if p.is_dir() and p.name.startswith(release_name)] if releases.exists() else []
    if len(matches) != 1:
        raise ValueError(f"release selector must match exactly one release: {release_name}")
    target = matches[0]
    manifest = _read_manifest(target)
    current = root / "current"
    if current.is_symlink():
        old = current.resolve(strict=True)
        if old != target:
            _symlink_atomic(root / "previous", old)
    _symlink_atomic(current, target)
    return manifest


def install_runtime(
    *,
    repo_root: str | os.PathLike[str],
    source_python: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str] | None = None,
    allow_unpushed: bool = False,
) -> OptionRuntimeRelease:
    repo = Path(repo_root).expanduser().resolve()
    source_python_path = Path(source_python).expanduser().absolute()
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    _tracked_clean(repo)
    sha, upstream_sha = _git_sha(repo)
    if not allow_unpushed and upstream_sha != sha:
        raise ValueError("HEAD must match its upstream before production runtime install")

    python_version, freeze = _freeze_environment(source_python_path, repo)
    fingerprint = dependency_fingerprint(python_version=python_version, freeze_text=freeze)
    venv = _ensure_runtime_env(
        runtime_root=root,
        repo_root=repo,
        source_python=source_python_path,
        fingerprint=fingerprint,
    )

    release_name = sha[:12]
    releases = root / "releases"
    release = releases / release_name
    if not release.exists():
        releases.mkdir(parents=True, exist_ok=True)
        temporary = releases / f".{release_name}.tmp-{os.getpid()}"
        if temporary.exists():
            _remove_tree(temporary)
        try:
            app = temporary / "app"
            _extract_git_head(repo, app)
            manifest = OptionRuntimeRelease(
                git_sha=sha,
                git_short=release_name,
                dependency_fingerprint=fingerprint,
                env_relpath=str(venv.relative_to(root)),
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python_version=python_version,
            )
            _write_json_atomic(temporary / _MANIFEST_NAME, asdict(manifest))
            _chmod_readonly_tree(app)
            os.replace(temporary, release)
        finally:
            if temporary.exists():
                _remove_tree(temporary)
    else:
        manifest = _read_manifest(release)
        if manifest.git_sha != sha or manifest.dependency_fingerprint != fingerprint:
            raise ValueError(f"existing release directory does not match requested runtime: {release}")

    base_python = Path(getattr(sys, "_base_executable", sys.executable)).expanduser().absolute()
    _write_runner(root, base_python)
    activate_release(root, release_name)
    return _read_manifest(release)


def list_releases(runtime_root: str | os.PathLike[str] | None = None) -> tuple[OptionRuntimeRelease, ...]:
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    releases = root / "releases"
    if not releases.exists():
        return ()
    found = []
    for item in releases.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            try:
                found.append(_read_manifest(item))
            except ValueError:
                continue
    return tuple(sorted(found, key=lambda release: release.created_at, reverse=True))


def rollback_runtime(
    runtime_root: str | os.PathLike[str] | None = None,
    *,
    to_release: str | None = None,
) -> OptionRuntimeRelease:
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    if to_release is not None:
        return activate_release(root, to_release)
    previous = root / "previous"
    if not previous.is_symlink():
        raise ValueError("no previous runtime release is available")
    return activate_release(root, previous.resolve(strict=True).name)


def health_runtime(runtime_root: str | os.PathLike[str] | None = None) -> OptionRuntimeHealth:
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    issues: list[str] = []
    runner = runtime_runner_path(root)
    current = root / "current"
    release_name = None
    manifest_path = None
    if not runner.exists() or not os.access(runner, os.X_OK):
        issues.append("runtime runner missing or not executable")
    if not current.is_symlink():
        issues.append("current release symlink is missing")
    else:
        try:
            release = current.resolve(strict=True)
            release_name = release.name
            manifest_path = str(release / _MANIFEST_NAME)
            manifest = _read_manifest(release)
            env_python = root / manifest.env_relpath / "bin" / "python"
            script = release / "app" / "scripts" / "options_daily_ops.py"
            if not env_python.exists():
                issues.append("runtime environment Python is missing")
            if not script.exists():
                issues.append("runtime daily-ops script is missing")
            if runner.exists() and not issues:
                result = subprocess.run(
                    [str(runner), "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    issues.append(f"runtime runner smoke failed with exit code {result.returncode}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            issues.append(f"runtime health error: {type(exc).__name__}")
    return OptionRuntimeHealth(
        status="PASS" if not issues else "FAIL",
        current_release=release_name,
        issues=tuple(issues),
        runner_path=str(runner),
        manifest_path=manifest_path,
    )


def prune_releases(
    runtime_root: str | os.PathLike[str] | None = None,
    *,
    keep: int = 3,
) -> tuple[str, ...]:
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be a positive whole number")
    root = Path(runtime_root).expanduser() if runtime_root is not None else default_runtime_root()
    protected: set[str] = set()
    for link_name in ("current", "previous"):
        link = root / link_name
        if link.is_symlink():
            protected.add(link.resolve(strict=True).name)
    releases = list(list_releases(root))
    keep_names = {release.git_short for release in releases[:keep]} | protected
    removed: list[str] = []
    for release in releases:
        if release.git_short in keep_names:
            continue
        path = root / "releases" / release.git_short
        _remove_tree(path)
        removed.append(release.git_short)
    return tuple(removed)
