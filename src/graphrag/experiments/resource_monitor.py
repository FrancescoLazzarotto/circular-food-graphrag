"""Background sampling of CPU, RAM and GPU usage during an experiment run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency fallback
    psutil = None


def _safe_float(value: str | None) -> float | None:
    """Parse a float, returning ``None`` for missing, ``N/A`` or bad values."""
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: str | None) -> int | None:
    """Like :func:`_safe_float`, truncated to an int."""
    parsed = _safe_float(value)
    return int(parsed) if parsed is not None else None


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty list."""
    return (sum(values) / len(values)) if values else None


def _max(values: list[float]) -> float | None:
    """Maximum, or ``None`` for an empty list."""
    return max(values) if values else None


def _read_meminfo_mb() -> tuple[float | None, float | None, float | None]:
    """Read system memory from ``/proc/meminfo``, for use without psutil.

    Returns:
        ``(used_mb, total_mb, used_percent)``, each ``None`` when unavailable.
    """
    mem_total_kb: float | None = None
    mem_available_kb: float | None = None

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_kb = _safe_float(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = _safe_float(line.split()[1])
                if mem_total_kb is not None and mem_available_kb is not None:
                    break
    except OSError:
        return None, None, None

    if mem_total_kb is None or mem_available_kb is None:
        return None, None, None

    used_mb = max(mem_total_kb - mem_available_kb, 0.0) / 1024.0
    total_mb = mem_total_kb / 1024.0
    percent = (used_mb / total_mb * 100.0) if total_mb > 0 else None
    return used_mb, total_mb, percent


def _read_proc_memory_mb() -> tuple[float | None, float | None]:
    """Read this process's memory from ``/proc/self/status``, without psutil.

    Returns:
        ``(rss_mb, vms_mb)``, each ``None`` when unavailable.
    """
    rss_kb: float | None = None
    vms_kb: float | None = None

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kb = _safe_float(line.split()[1])
                elif line.startswith("VmSize:"):
                    vms_kb = _safe_float(line.split()[1])
                if rss_kb is not None and vms_kb is not None:
                    break
    except OSError:
        return None, None

    rss_mb = (rss_kb / 1024.0) if rss_kb is not None else None
    vms_mb = (vms_kb / 1024.0) if vms_kb is not None else None
    return rss_mb, vms_mb


@dataclass
class GPUSample:
    """One ``nvidia-smi`` reading of one GPU.

    Attributes:
        index: GPU index.
        name: GPU model name.
        utilization_gpu_percent: Compute utilisation.
        utilization_memory_percent: Memory controller utilisation.
        memory_used_mb: Memory in use.
        memory_total_mb: Total memory.
        temperature_c: Temperature in degrees Celsius.
    """

    index: int
    name: str
    utilization_gpu_percent: float | None = None
    utilization_memory_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None


@dataclass
class ResourceSample:
    """One reading of system, process and GPU usage.

    Attributes:
        ts_utc: ISO timestamp in UTC.
        elapsed_sec: Seconds since the monitor started.
        system_cpu_percent: System-wide CPU usage (psutil only).
        system_ram_used_mb: System memory in use.
        system_ram_total_mb: Total system memory.
        system_ram_percent: System memory in use, in percent.
        process_cpu_percent: CPU usage of this process (psutil only).
        process_rss_mb: Resident memory of this process.
        process_vms_mb: Virtual memory of this process.
        process_num_threads: Thread count of this process (psutil only).
        gpus: One reading per GPU.
    """

    ts_utc: str
    elapsed_sec: float
    system_cpu_percent: float | None
    system_ram_used_mb: float | None
    system_ram_total_mb: float | None
    system_ram_percent: float | None
    process_cpu_percent: float | None
    process_rss_mb: float | None
    process_vms_mb: float | None
    process_num_threads: int | None
    gpus: list[GPUSample] = field(default_factory=list)


class ResourceMonitor:
    """Background sampler for CPU, RAM and GPU utilization."""

    def __init__(
        self, sample_interval_sec: float = 1.0, include_gpu: bool = True
    ) -> None:
        """Create a stopped monitor.

        Args:
            sample_interval_sec: Seconds between samples.
            include_gpu: Sample GPUs through ``nvidia-smi`` when it is on PATH.

        Raises:
            ValueError: If ``sample_interval_sec`` is not positive.
        """
        if sample_interval_sec <= 0:
            raise ValueError("sample_interval_sec must be > 0")

        self.sample_interval_sec = sample_interval_sec
        self.include_gpu = include_gpu

        self._nvidia_smi = shutil.which("nvidia-smi") if include_gpu else None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._samples: list[ResourceSample] = []
        self._started_utc: str | None = None
        self._finished_utc: str | None = None
        self._start_monotonic: float | None = None

    def start(self) -> None:
        """Take a first sample and start the sampling thread; no-op if running."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._finished_utc = None
        self._start_monotonic = time.monotonic()

        if psutil is not None:
            psutil.cpu_percent(interval=None)
        if self._process is not None:
            self._process.cpu_percent(interval=None)

        self.capture_sample()
        self._thread = threading.Thread(
            target=self._sampling_loop, name="resource-monitor", daemon=True
        )
        self._thread.start()

    def _sampling_loop(self) -> None:
        """Capture a sample every interval until stopped."""
        while not self._stop_event.wait(self.sample_interval_sec):
            self.capture_sample()

    def stop(self) -> dict[str, Any]:
        """Stop sampling, take a final sample and summarise.

        Returns:
            The :meth:`summary` of all samples.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.sample_interval_sec * 2.0))

        self.capture_sample()
        self._finished_utc = datetime.now(timezone.utc).isoformat()
        return self.summary()

    def capture_sample(self) -> ResourceSample:
        """Read current usage and append it to the samples.

        Returns:
            The new sample.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        elapsed_sec = 0.0
        if self._start_monotonic is not None:
            elapsed_sec = max(time.monotonic() - self._start_monotonic, 0.0)

        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
            virtual_memory = psutil.virtual_memory()
            system_ram_used_mb = float(virtual_memory.used) / (1024.0 * 1024.0)
            system_ram_total_mb = float(virtual_memory.total) / (1024.0 * 1024.0)
            system_ram_percent = float(virtual_memory.percent)
        else:
            cpu_percent = None
            system_ram_used_mb, system_ram_total_mb, system_ram_percent = (
                _read_meminfo_mb()
            )

        if self._process is not None:
            process_cpu_percent = self._process.cpu_percent(interval=None)
            memory_info = self._process.memory_info()
            process_rss_mb = float(memory_info.rss) / (1024.0 * 1024.0)
            process_vms_mb = float(memory_info.vms) / (1024.0 * 1024.0)
            process_num_threads = int(self._process.num_threads())
        else:
            process_cpu_percent = None
            process_rss_mb, process_vms_mb = _read_proc_memory_mb()
            process_num_threads = None

        sample = ResourceSample(
            ts_utc=now_utc,
            elapsed_sec=elapsed_sec,
            system_cpu_percent=cpu_percent,
            system_ram_used_mb=system_ram_used_mb,
            system_ram_total_mb=system_ram_total_mb,
            system_ram_percent=system_ram_percent,
            process_cpu_percent=process_cpu_percent,
            process_rss_mb=process_rss_mb,
            process_vms_mb=process_vms_mb,
            process_num_threads=process_num_threads,
            gpus=self._collect_gpu_samples(),
        )

        with self._lock:
            self._samples.append(sample)

        return sample

    def _collect_gpu_samples(self) -> list[GPUSample]:
        """Query ``nvidia-smi``; empty when unavailable or when the call fails."""
        if not self._nvidia_smi:
            return []

        command = [
            self._nvidia_smi,
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception:
            return []

        gpu_samples: list[GPUSample] = []
        for raw_line in completed.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue

            index = _safe_int(parts[0])
            if index is None:
                continue

            gpu_samples.append(
                GPUSample(
                    index=index,
                    name=parts[1],
                    utilization_gpu_percent=_safe_float(parts[2]),
                    utilization_memory_percent=_safe_float(parts[3]),
                    memory_used_mb=_safe_float(parts[4]),
                    memory_total_mb=_safe_float(parts[5]),
                    temperature_c=_safe_float(parts[6]),
                )
            )

        return gpu_samples

    def samples(self) -> list[ResourceSample]:
        """Return a copy of the samples taken so far."""
        with self._lock:
            return list(self._samples)

    def summary(self) -> dict[str, Any]:
        """Aggregate the samples.

        Returns:
            Start/finish times, duration, sample count, availability flags,
            average and peak system and process usage, and per-GPU average and
            peak utilisation, memory and temperature. Metrics without data are
            ``None``.
        """
        samples = self.samples()

        system_cpu = [
            s.system_cpu_percent for s in samples if s.system_cpu_percent is not None
        ]
        system_ram_pct = [
            s.system_ram_percent for s in samples if s.system_ram_percent is not None
        ]
        system_ram_used = [
            s.system_ram_used_mb for s in samples if s.system_ram_used_mb is not None
        ]
        process_cpu = [
            s.process_cpu_percent for s in samples if s.process_cpu_percent is not None
        ]
        process_rss = [
            s.process_rss_mb for s in samples if s.process_rss_mb is not None
        ]
        process_vms = [
            s.process_vms_mb for s in samples if s.process_vms_mb is not None
        ]

        duration_sec = samples[-1].elapsed_sec if samples else 0.0

        gpu_aggregate: dict[int, dict[str, Any]] = {}
        for sample in samples:
            for gpu in sample.gpus:
                entry = gpu_aggregate.setdefault(
                    gpu.index,
                    {
                        "index": gpu.index,
                        "name": gpu.name,
                        "utilization_gpu_percent_values": [],
                        "utilization_memory_percent_values": [],
                        "memory_used_mb_values": [],
                        "temperature_c_values": [],
                        "memory_total_mb": gpu.memory_total_mb,
                    },
                )

                if gpu.memory_total_mb is not None:
                    entry["memory_total_mb"] = gpu.memory_total_mb

                if gpu.utilization_gpu_percent is not None:
                    entry["utilization_gpu_percent_values"].append(
                        gpu.utilization_gpu_percent
                    )
                if gpu.utilization_memory_percent is not None:
                    entry["utilization_memory_percent_values"].append(
                        gpu.utilization_memory_percent
                    )
                if gpu.memory_used_mb is not None:
                    entry["memory_used_mb_values"].append(gpu.memory_used_mb)
                if gpu.temperature_c is not None:
                    entry["temperature_c_values"].append(gpu.temperature_c)

        gpu_summary: list[dict[str, Any]] = []
        for gpu_index in sorted(gpu_aggregate):
            entry = gpu_aggregate[gpu_index]
            gpu_summary.append(
                {
                    "index": entry["index"],
                    "name": entry["name"],
                    "memory_total_mb": entry["memory_total_mb"],
                    "avg_utilization_gpu_percent": _mean(
                        entry["utilization_gpu_percent_values"]
                    ),
                    "peak_utilization_gpu_percent": _max(
                        entry["utilization_gpu_percent_values"]
                    ),
                    "avg_utilization_memory_percent": _mean(
                        entry["utilization_memory_percent_values"]
                    ),
                    "peak_utilization_memory_percent": _max(
                        entry["utilization_memory_percent_values"]
                    ),
                    "avg_memory_used_mb": _mean(entry["memory_used_mb_values"]),
                    "peak_memory_used_mb": _max(entry["memory_used_mb_values"]),
                    "avg_temperature_c": _mean(entry["temperature_c_values"]),
                    "peak_temperature_c": _max(entry["temperature_c_values"]),
                }
            )

        return {
            "monitoring_started_utc": self._started_utc,
            "monitoring_finished_utc": self._finished_utc,
            "monitoring_duration_sec": duration_sec,
            "sample_count": len(samples),
            "sample_interval_sec": self.sample_interval_sec,
            "psutil_available": psutil is not None,
            "gpu_monitoring_enabled": self.include_gpu,
            "gpu_sampling_available": bool(self._nvidia_smi),
            "avg_system_cpu_percent": _mean(system_cpu),
            "peak_system_cpu_percent": _max(system_cpu),
            "avg_system_ram_percent": _mean(system_ram_pct),
            "peak_system_ram_percent": _max(system_ram_pct),
            "peak_system_ram_used_mb": _max(system_ram_used),
            "avg_process_cpu_percent": _mean(process_cpu),
            "peak_process_cpu_percent": _max(process_cpu),
            "peak_process_rss_mb": _max(process_rss),
            "peak_process_vms_mb": _max(process_vms),
            "gpus": gpu_summary,
        }

    def export_samples_jsonl(self, path: str) -> None:
        """Write one JSON line per sample to ``path``, creating its directory."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", encoding="utf-8") as output_file:
            for sample in self.samples():
                output_file.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

    def export_summary_json(
        self, path: str, extra: dict[str, Any] | None = None
    ) -> None:
        """Write the :meth:`summary` as JSON to ``path``.

        Args:
            path: Output file; its directory is created.
            extra: Keys merged into the summary before writing.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        data = self.summary()
        if extra:
            data.update(extra)

        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
