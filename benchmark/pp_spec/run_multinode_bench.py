#!/usr/bin/env python3
"""Run the two-node/four-GPU Qwen3.5-27B SPECTRE sweep.

Rank 0 is launched directly on this host. Rank 1 is launched in an existing
hostNetwork/hostPID Kubernetes maintenance pod on node s10, through ``/host``
and as the host's ``liyi`` user. The runner never creates or modifies a
Kubernetes resource.

"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

MODEL = "Qwen/Qwen3.5-27B"
DRAFT_MODEL = "z-lab/Qwen3.5-27B-DFlash"
NUM_LAYERS = 64
UNIFORM_PARTITION = (16, 16, 16, 16)
LOAD_POINTS_TEXT = (
    "8:2:32,16:4:64,32:8:128,64:16:256,96:24:384,128:32:512"
)
MEM_FRACTION_STATIC = 0.75
DFLASH_BLOCK_SIZE = 16
MAMBA_SSM_DTYPE = "float32"
MAMBA_FULL_MEMORY_RATIO = 0.9
MODEL_DTYPE = "bfloat16"
MAX_TOKENS = 1024
PROMPT_MAX_TOKENS = 2000
LOCAL_GPU_IDS = (0, 1)
REMOTE_GPU_IDS = (0, 1)
NETWORK_INTERFACE = "ibp65s0f0"
SYSTEM_PATH_SUFFIX = (
    "/usr/local/cuda-13.2/bin",
    "/usr/local/cuda/bin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)

REMOTE_START_SCRIPT = r"""
set -euo pipefail
repo_root=$1
server_bin=$2
pid_file=$3
log_file=$4
gpu_ids=$5
shift 5

mkdir -p -- "$(dirname -- "$pid_file")" "$(dirname -- "$log_file")"
cd -- "$repo_root"
nohup setsid /usr/bin/env \
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
  NCCL_SOCKET_IFNAME=""" + NETWORK_INTERFACE + r""" \
  GLOO_SOCKET_IFNAME=""" + NETWORK_INTERFACE + r""" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  SGLANG_RAGGED_VERIFY_MODE=static \
  SGL_FORCE_SHUTDOWN=1 \
  PYTHONUNBUFFERED=1 \
  "$server_bin" "$@" </dev/null >"$log_file" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
sleep 1
kill -0 "$pid"
printf 'REMOTE_PROCESS\t%s\n' "$pid"
"""

REMOTE_STOP_SCRIPT = r"""
set -euo pipefail
pid_file=$1
int_wait=$2
term_wait=$3
[[ -f $pid_file ]] || exit 0
pid=$(<"$pid_file")

group_alive() {
  kill -0 -- "-$pid" 2>/dev/null
}

wait_group() {
  local deadline=$((SECONDS + $1))
  while group_alive && ((SECONDS < deadline)); do
    sleep 1
  done
  ! group_alive
}

if group_alive; then
  kill -INT -- "-$pid" 2>/dev/null || true
  if ! wait_group "$int_wait"; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    if ! wait_group "$term_wait"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      sleep 1
    fi
  fi
fi
rm -f -- "$pid_file"
"""


class RunnerError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class LoadPoint:
    concurrency: int
    qps: int
    requests: int

    @property
    def text(self) -> str:
        return f"{self.concurrency}:{self.qps}:{self.requests}"

    @property
    def tag(self) -> str:
        return f"c{self.concurrency}_qps{self.qps}_n{self.requests}"


@dataclasses.dataclass(frozen=True)
class Topology:
    name: str
    tp_size: int
    pp_size: int
    partition: tuple[int, ...] | None

    def graph_batch_size(self, concurrency: int) -> int:
        if self.pp_size == 1:
            return concurrency
        if concurrency % self.pp_size:
            raise RunnerError(
                f"concurrency {concurrency} is not divisible by PP={self.pp_size}"
            )
        return concurrency // self.pp_size


@dataclasses.dataclass
class RemoteProcess:
    pid_file: str
    log_file: str


@dataclasses.dataclass
class LocalProcess:
    process: subprocess.Popen[bytes]
    log_handle: Any


def parse_load_points(text: str = LOAD_POINTS_TEXT) -> tuple[LoadPoint, ...]:
    points: list[LoadPoint] = []
    for raw in text.split(","):
        match = re.fullmatch(r"([1-9][0-9]*):([1-9][0-9]*):([1-9][0-9]*)", raw)
        if not match:
            raise RunnerError(f"invalid built-in load point: {raw!r}")
        points.append(LoadPoint(*(int(value) for value in match.groups())))
    return tuple(points)


def parse_partition(value: str) -> tuple[int, ...]:
    try:
        partition = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "partition must be four comma-separated positive integers"
        ) from exc
    if len(partition) != 4 or any(item <= 0 for item in partition):
        raise argparse.ArgumentTypeError(
            "partition must be four comma-separated positive integers"
        )
    if sum(partition) != NUM_LAYERS:
        raise argparse.ArgumentTypeError(
            f"partition must cover {NUM_LAYERS} layers; got sum={sum(partition)}"
        )
    return partition


def controlled_path(repo_root: str | Path) -> str:
    return ":".join((f"{repo_root}/.venv/bin", *SYSTEM_PATH_SUFFIX))


def run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    text: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            text=text,
            capture_output=capture_output,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerError(
            f"command timed out after {timeout}s: {shlex.join(command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() if text else ""
        stdout = (exc.stdout or "").strip() if text else ""
        detail = stderr or stdout or f"exit code {exc.returncode}"
        raise RunnerError(f"command failed: {shlex.join(command)}\n{detail}") from exc


def topology_map(auto_partition: tuple[int, ...] | None) -> dict[str, Topology]:
    return {
        "tp4": Topology("tp4", tp_size=4, pp_size=1, partition=None),
        "pp4_uniform": Topology(
            "pp4_uniform", tp_size=1, pp_size=4, partition=UNIFORM_PARTITION
        ),
        "pp4_auto": Topology(
            "pp4_auto", tp_size=1, pp_size=4, partition=auto_partition
        ),
    }


class MultiNodeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = args.repo_root.resolve()
        self.script_dir = self.repo_root / "benchmark" / "pp_spec"
        self.dataset = args.dataset.resolve()
        self.output_root = args.output_root.expanduser().resolve()
        self.server_bin = self.repo_root / ".venv" / "bin" / "sglang"
        self.python_bin = self.repo_root / ".venv" / "bin" / "python"
        self.remote_repo_root = args.remote_repo_root
        self.remote_server_bin = f"{self.remote_repo_root}/.venv/bin/sglang"
        self.local_ip = args.local_ip or self._route_source()
        dist_host = f"[{self.local_ip}]" if ":" in self.local_ip else self.local_ip
        self.dist_init_addr = f"{dist_host}:{args.dist_port}"
        all_topologies = topology_map(args.auto_partition)
        self.topologies = [all_topologies[name] for name in args.topology]
        self.points = parse_load_points()
        self.current_local: LocalProcess | None = None
        self.current_remote: RemoteProcess | None = None

    def _route_source(self) -> str:
        result = run_checked(
            ["ip", "-4", "-o", "address", "show", "dev", NETWORK_INTERFACE]
        )
        match = re.search(r"\binet\s+(\d+(?:\.\d+){3})/", result.stdout)
        if not match:
            raise RunnerError(
                f"cannot determine IPv4 address for {NETWORK_INTERFACE}; "
                "pass --local-ip"
            )
        return match.group(1)

    def remote_user_command(self, script: str, script_args: Sequence[str]) -> list[str]:
        remote_path = controlled_path(self.remote_repo_root)
        kubectl = ["sudo", "kubectl"]
        if self.args.kubectl_context:
            kubectl.extend(["--context", self.args.kubectl_context])
        return [
            *kubectl,
            "exec",
            "--namespace",
            self.args.namespace,
            self.args.worker_pod,
            "--",
            "chroot",
            "/host",
            "/usr/bin/prlimit",
            "--memlock=unlimited:unlimited",
            "--",
            "/usr/bin/setpriv",
            f"--reuid={self.args.remote_user}",
            f"--regid={self.args.remote_user}",
            "--init-groups",
            "--reset-env",
            "/usr/bin/env",
            f"PATH={remote_path}",
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            script,
            "pp-spec-remote",
            *script_args,
        ]

    def local_server_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, LOCAL_GPU_IDS)),
            "NCCL_SOCKET_IFNAME": NETWORK_INTERFACE,
            "GLOO_SOCKET_IFNAME": NETWORK_INTERFACE,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SGLANG_RAGGED_VERIFY_MODE": "static",
            "SGL_FORCE_SHUTDOWN": "1",
            "PYTHONUNBUFFERED": "1",
            }
        )
        return environment

    def server_args(
        self, topology: Topology, point: LoadPoint, node_rank: int
    ) -> list[str]:
        graph_bs = topology.graph_batch_size(point.concurrency)
        command = [
            "serve",
            "--model-path",
            MODEL,
            "--revision",
            self.args.model_revision,
            "--tp-size",
            str(topology.tp_size),
            "--pp-size",
            str(topology.pp_size),
            "--dp-size",
            "1",
            "--nnodes",
            "2",
            "--node-rank",
            str(node_rank),
            "--dist-init-addr",
            self.dist_init_addr,
            "--base-gpu-id",
            "0",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            DRAFT_MODEL,
            "--speculative-draft-model-revision",
            self.args.draft_model_revision,
            "--speculative-draft-attention-backend",
            "flashinfer",
            "--speculative-dflash-block-size",
            str(DFLASH_BLOCK_SIZE),
            "--dtype",
            MODEL_DTYPE,
            "--attention-backend",
            "flashinfer",
            "--linear-attn-backend",
            "triton",
            "--mamba-ssm-dtype",
            MAMBA_SSM_DTYPE,
            "--mamba-full-memory-ratio",
            str(MAMBA_FULL_MEMORY_RATIO),
            "--enable-linear-replayssm-spec",
            "--max-running-requests",
            str(point.concurrency),
            "--cuda-graph-max-bs-decode",
            str(graph_bs),
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
            "--page-size",
            "1",
            "--random-seed",
            "1",
            "--disable-radix-cache",
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.args.port),
        ]
        if topology.pp_size > 1:
            if topology.partition is None:
                raise RunnerError(f"{topology.name} requires a partition")
            command.extend(
                ["--pp-layer-partition", ",".join(map(str, topology.partition))]
            )
            command.append("--disable-overlap-schedule")
        return command

    def benchmark_args(
        self, topology: Topology, point: LoadPoint, output_dir: Path
    ) -> list[str]:
        return [
            str(self.python_bin),
            str(self.script_dir / "bench_spectre.py"),
            "--url",
            f"http://127.0.0.1:{self.args.port}",
            "--label",
            topology.name,
            "--dataset",
            str(self.dataset),
            "--tokenizer",
            MODEL,
            "--tokenizer-revision",
            self.args.model_revision,
            "--load-points",
            point.text,
            "--max-tokens",
            str(MAX_TOKENS),
            "--prompt-max-tokens",
            str(PROMPT_MAX_TOKENS),
            "--temperature",
            "0",
            "--seed",
            "1",
            "--request-timeout-s",
            str(self.args.request_timeout),
            "--cooldown-s",
            str(self.args.cooldown),
            "--output-dir",
            str(output_dir),
        ]

    def launch_local(
        self, topology: Topology, point: LoadPoint, point_dir: Path
    ) -> None:
        if self.current_local is not None:
            raise RunnerError("local rank is already active")
        log_handle = (point_dir / "server_node0.log").open("wb", buffering=0)
        command = [str(self.server_bin), *self.server_args(topology, point, 0)]
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=self.local_server_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_handle.close()
            raise
        self.current_local = LocalProcess(process, log_handle)

    def launch_remote(
        self,
        topology: Topology,
        point: LoadPoint,
        point_dir: Path,
        token: str,
    ) -> None:
        if self.current_remote is not None:
            raise RunnerError("remote rank is already active")
        pid_file = f"/tmp/sglang-pp-spec/{token}.pid"
        log_file = f"/tmp/sglang-pp-spec/{token}.log"
        self.current_remote = RemoteProcess(pid_file, log_file)
        command = self.remote_user_command(
            REMOTE_START_SCRIPT,
            [
                self.remote_repo_root,
                self.remote_server_bin,
                pid_file,
                log_file,
                ",".join(map(str, REMOTE_GPU_IDS)),
                *self.server_args(topology, point, 1),
            ],
        )
        try:
            run_checked(command, timeout=self.args.kubectl_timeout)
        except BaseException:
            try:
                self.stop_remote()
            except BaseException:
                pass
            try:
                self.collect_remote_log(point_dir / "server_node1.log", log_file)
            except BaseException:
                pass
            raise

    def wait_for_server(self) -> None:
        local = self.current_local
        if local is None:
            raise RunnerError("local rank was not launched")
        url = f"http://127.0.0.1:{self.args.port}/model_info"
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            code = local.process.poll()
            if code is not None:
                raise RunnerError(
                    f"local rank exited during startup with code {code}; "
                    "see server_node0.log"
                )
            response = subprocess.run(
                ["curl", "-fsS", "--max-time", "2", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if response.returncode == 0:
                return
            time.sleep(1)
        raise RunnerError(f"server startup timed out after {self.args.startup_timeout}s")

    def begin_local_shutdown(self) -> None:
        local = self.current_local
        if local is None:
            return
        try:
            os.killpg(local.process.pid, signal.SIGINT)
        except ProcessLookupError:
            return

    def stop_remote_by_file(self, pid_file: str) -> None:
        run_checked(
            self.remote_user_command(
                REMOTE_STOP_SCRIPT,
                [
                    pid_file,
                    str(self.args.interrupt_timeout),
                    str(self.args.terminate_timeout),
                ],
            ),
            timeout=(
                self.args.kubectl_timeout
                + self.args.interrupt_timeout
                + self.args.terminate_timeout
                + 15
            ),
        )

    def stop_remote(self) -> None:
        remote = self.current_remote
        if remote is None:
            return
        self.stop_remote_by_file(remote.pid_file)
        self.current_remote = None

    def finish_local_shutdown(self) -> None:
        local = self.current_local
        if local is None:
            return
        try:
            try:
                local.process.wait(timeout=self.args.interrupt_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(local.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    local.process.wait(timeout=self.args.terminate_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(local.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    local.process.wait(timeout=10)
        finally:
            if not local.log_handle.closed:
                local.log_handle.close()
            self.current_local = None

    def collect_remote_log(self, destination: Path, remote_log: str) -> None:
        command = self.remote_user_command(
            'set -euo pipefail; exec /bin/cat -- "$1"', [remote_log]
        )
        result = run_checked(
            command, timeout=self.args.kubectl_timeout, text=False
        )
        destination.write_bytes(result.stdout)
        run_checked(
            self.remote_user_command(
                "set -euo pipefail; file=$1; "
                '[[ $file == /tmp/sglang-pp-spec/* ]]; rm -f -- "$file"',
                [remote_log],
            ),
            timeout=self.args.kubectl_timeout,
        )

    def cleanup_point(self, point_dir: Path) -> None:
        remote = self.current_remote
        errors: list[str] = []
        try:
            self.stop_remote()
        except BaseException as exc:
            errors.append(f"remote cleanup failed: {exc}")
        try:
            self.begin_local_shutdown()
            self.finish_local_shutdown()
        except BaseException as exc:
            errors.append(f"local cleanup failed: {exc}")
        if remote is not None and self.current_remote is None:
            try:
                self.collect_remote_log(
                    point_dir / "server_node1.log", remote.log_file
                )
            except BaseException as exc:
                errors.append(f"remote log collection failed: {exc}")
        if errors:
            raise RunnerError("; ".join(errors))

    def run_point(self, topology: Topology, point: LoadPoint) -> None:
        point_dir = self.output_root / topology.name / point.tag
        point_dir.mkdir(parents=True, exist_ok=True)
        token = (
            f"ppspec-{topology.name}-{point.tag}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:10]}"
        )
        print(
            f"[{topology.name}][{point.text}] start: "
            f"max_running={point.concurrency} "
            f"graph={topology.graph_batch_size(point.concurrency)}",
            flush=True,
        )
        primary_error: BaseException | None = None
        try:
            self.launch_local(topology, point, point_dir)
            self.launch_remote(topology, point, point_dir, token)
            self.wait_for_server()
            run_checked(
                self.benchmark_args(topology, point, point_dir),
                cwd=self.repo_root,
                capture_output=False,
            )
        except BaseException as exc:
            primary_error = exc
        try:
            self.cleanup_point(point_dir)
        except BaseException as cleanup_exc:
            if primary_error is None:
                primary_error = cleanup_exc
            else:
                print(f"[cleanup] {cleanup_exc}", file=sys.stderr, flush=True)
        if primary_error is not None:
            raise primary_error
        print(f"[{topology.name}][{point.text}] complete", flush=True)

    def finalize(self) -> None:
        run_checked(
            [
                str(self.python_bin),
                str(self.script_dir / "summarize_spectre.py"),
                str(self.output_root),
            ],
            cwd=self.repo_root,
            capture_output=False,
        )
        run_checked(
            [
                str(self.python_bin),
                str(self.script_dir / "plot_performance.py"),
                str(self.output_root / "summary.csv"),
            ],
            cwd=self.repo_root,
            capture_output=False,
        )

    def run(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        print(f"Results: {self.output_root}", flush=True)
        for topology in self.topologies:
            for point in self.points:
                self.run_point(topology, point)
        self.finalize()
        print(f"All benchmark points complete: {self.output_root}", flush=True)


def default_output_root() -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "results" / f"Qwen_Qwen3.5-27B_multinode_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-pod", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--kubectl-context")
    parser.add_argument("--remote-user", default="liyi")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--draft-model-revision", default="main")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--remote-repo-root", default=str(REPO_ROOT))
    parser.add_argument("--dataset", type=Path, default=SCRIPT_DIR / "data" / "sharegpt.json")
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument(
        "--topology",
        action="append",
        choices=("tp4", "pp4_uniform", "pp4_auto"),
        help="topology to run; repeat as needed (default: all three)",
    )
    parser.add_argument(
        "--auto-partition",
        type=parse_partition,
        help="non-uniform PP partition as four layer counts totaling 64",
    )
    parser.add_argument(
        "--local-ip",
        help=f"rank-0 address on {NETWORK_INTERFACE} (default: interface IPv4)",
    )
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--dist-port", type=int, default=29500)
    parser.add_argument(
        "--kubectl-timeout",
        type=float,
        default=30.0,
        help="timeout in seconds for each kubectl API/exec operation",
    )
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--interrupt-timeout", type=int, default=60)
    parser.add_argument("--terminate-timeout", type=int, default=10)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    args.topology = list(
        dict.fromkeys(args.topology or ["tp4", "pp4_uniform", "pp4_auto"])
    )
    if "pp4_auto" in args.topology and args.auto_partition is None:
        parser.error("--auto-partition is required when pp4_auto is selected")
    if "pp4_auto" in args.topology and args.auto_partition == UNIFORM_PARTITION:
        parser.error("--auto-partition must be non-uniform for pp4_auto")
    if not 1 <= args.port <= 65535 or not 1 <= args.dist_port <= 65535:
        parser.error("--port and --dist-port must be in [1, 65535]")
    if args.port == args.dist_port:
        parser.error("--port and --dist-port must differ")
    for name in (
        "startup_timeout",
        "request_timeout",
        "kubectl_timeout",
        "interrupt_timeout",
        "terminate_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.cooldown < 0:
        parser.error("--cooldown must be non-negative")
    if not args.remote_repo_root.startswith("/") or "\n" in args.remote_repo_root:
        parser.error("--remote-repo-root must be an absolute single-line path")
    for name in ("model_revision", "draft_model_revision"):
        value = getattr(args, name)
        if not value or any(character.isspace() for character in value):
            parser.error(f"--{name.replace('_', '-')} must be a non-empty token")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for shutdown_signal in (
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
    ):
        signal.signal(shutdown_signal, request_shutdown)
    try:
        MultiNodeRunner(args).run()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except RunnerError as exc:
        raise SystemExit(f"[multinode] {exc}") from exc


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
    except BrokenPipeError:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_fd, sys.stdout.fileno())
        finally:
            os.close(null_fd)
        raise SystemExit(0)
