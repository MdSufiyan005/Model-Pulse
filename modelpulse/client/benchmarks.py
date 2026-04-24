"""
Edge-aware, benchmark suite for inference testing and metric aggregation.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from modelpulse.shared.models import InferenceMetrics

#  Benchmark profiles  

BENCHMARK_PROFILES: dict[str, int] = {
    "light":  64,
    "medium": 128,
    "heavy":  256,
}

# Per-question token budgets
# Keyed by exact question text.  When a question is not found here the
# suite falls back to the profile default.  Budgets are chosen to be large
# enough for a complete answer but small enough to avoid padding the latency
# numbers with repetition / padding tokens.

QUESTION_MAX_TOKENS: dict[str, int] = {
    "What is 15 * 12?":                         16,   # one-line arithmetic
    "What is the capital of France?":            16,   # one-word answer
    "Explain machine learning in one sentence.": 64,   # one sentence
    "List 3 uses of a REST API.":                96,   # three bullet points
    "Write a 3-word haiku about spring.":        24,   # three words
    "What is a hash function?":                  96,   # short definition
}

# Questions 

BENCHMARK_QUESTIONS = list(QUESTION_MAX_TOKENS.keys())

#  Thermal throttle threshold 
THROTTLE_WARN_C: float = 80.0

#   Results  

@dataclass
class QuestionResult:
    """Per-question outcome, including truncation detection."""
    index: int
    question: str
    tokens_generated: int
    tokens_per_sec: float
    time_to_first_tok_s: float
    latency_s: float
    max_tokens_used: int
    truncated: bool          # tokens_generated >= max_tokens_used - 1
    timed_out: bool = False
    error: bool = False


@dataclass
class BenchmarkResults:
    total_time_s: float = 0.0
    warmup_time_s: float = 0.0
    load_time_s: float = 0.0
    inference_time_s: float = 0.0

    tokens_per_sec: dict[str, float] = field(default_factory=dict)
    total_tokens_generated: int = 0
    avg_tokens_per_question: float = 0.0
    # total_tokens / total_inference_time (not mean of rates)
    avg_tokens_per_sec: float = 0.0
    # Same but computed from non-truncated questions only
    avg_tokens_per_sec_clean: Optional[float] = None
    perplexity: Optional[float] = None

    ttft_values: list[float] = field(default_factory=list)
    avg_ttft_s: float = 0.0
    min_ttft_s: float = 0.0
    max_ttft_s: float = 0.0

    latency_values: list[float] = field(default_factory=list)
    avg_latency_s: float = 0.0
    median_latency_s: float = 0.0
    p95_latency_s: float = 0.0

    ram_delta_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0          # system total RAM for utilisation %

    cpu_temp_c: Optional[float] = None
    avg_cpu_percent: float = 0.0
    thermal_throttle_warning: bool = False
    device_hw: str = ""
    os_info: str = ""

    question_count: int = 0
    output_lengths: list[int] = field(default_factory=list)
    question_results: list[QuestionResult] = field(default_factory=list)
    truncated_count: int = 0

    success_count: int = 0
    fail_count: int = 0

    server_url: str = ""
    source_model: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_inference_metrics(self) -> InferenceMetrics:
        return InferenceMetrics(
            load_time_s         = self.load_time_s,
            time_to_first_tok_s = self.avg_ttft_s,
            tokens_per_sec      = self.avg_tokens_per_sec,
            tokens_generated    = self.total_tokens_generated,
            ram_delta_mb        = self.ram_delta_mb,
            ram_used_mb         = self.ram_used_mb,
            cpu_temp_c          = self.cpu_temp_c,
            cpu_percent         = self.avg_cpu_percent,
            device_hw           = self.device_hw,
            os_info             = self.os_info,
            perplexity          = self.perplexity,
            timestamp           = self.timestamp,
            prompt              = f"Benchmark suite ({self.question_count} questions)",
            output              = (
                f"Processed {self.total_tokens_generated} tokens "
                f"across {self.question_count} questions"
            ),
            server_url   = self.server_url,
            source_model = self.source_model,
        )


#  Helpers  

def _percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    values = sorted(data)
    idx = max(0, min(int(round((len(values) - 1) * (p / 100.0))), len(values) - 1))
    return values[idx]


def _system_ram_total_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1_048_576
    except Exception:
        return 0.0


def _is_truncated(tokens_generated: int, max_tokens: int) -> bool:
    """
    A response is considered truncated when the model hit the token budget
    before naturally finishing.  We use max_tokens - 1 as the threshold
    because llama.cpp may emit exactly max_tokens or one fewer depending on
    how it counts the EOS token.
    """
    return tokens_generated >= max(1, max_tokens - 1)


#  Aggregate

def aggregate_metrics(
    metrics_list: list[InferenceMetrics],
    question_results: list[QuestionResult],
    load_time_s: float,
    latency_values: Optional[list[float]] = None,
) -> BenchmarkResults:
    """
    Build BenchmarkResults from per-question InferenceMetrics + QuestionResults.

    Parameters
    metrics_list      Per-question InferenceMetrics from ShardBridge.infer().
    question_results  Per-question QuestionResult (truncation flags etc.).
    load_time_s       From ShardBridge._load_time_s.
    latency_values    Wall-clock times from asyncio (preferred over reconstruction).
    """
    if not metrics_list:
        return BenchmarkResults(load_time_s=load_time_s)

    result = BenchmarkResults(
        load_time_s      = load_time_s,
        question_count   = len(metrics_list),
        source_model     = metrics_list[0].source_model,
        device_hw        = metrics_list[0].device_hw,
        os_info          = metrics_list[0].os_info,
        cpu_temp_c       = metrics_list[0].cpu_temp_c,
        server_url       = metrics_list[0].server_url,
        timestamp        = metrics_list[0].timestamp,
        question_results = question_results,
        ram_total_mb     = _system_ram_total_mb(),
    )

    total_tokens: int         = 0
    ttft_values: list[float]  = []
    cpu_percents: list[float] = []
    output_lengths: list[int] = []

    for i, m in enumerate(metrics_list):
        total_tokens += m.tokens_generated
        output_lengths.append(len(m.output))
        result.tokens_per_sec[f"q{i + 1}"] = m.tokens_per_sec
        if m.time_to_first_tok_s > 0:
            ttft_values.append(m.time_to_first_tok_s)
        if m.cpu_percent > 0:
            cpu_percents.append(m.cpu_percent)

    result.total_tokens_generated  = total_tokens
    result.avg_tokens_per_question = total_tokens / len(metrics_list)
    result.output_lengths          = output_lengths
    result.ram_delta_mb            = metrics_list[-1].ram_delta_mb
    result.ram_used_mb             = metrics_list[-1].ram_used_mb
    result.truncated_count         = sum(1 for qr in question_results if qr.truncated)

    # Latency
    if latency_values and len(latency_values) == len(metrics_list):
        lat = latency_values
    else:
        lat = []
        for m in metrics_list:
            if m.tokens_per_sec > 0 and m.tokens_generated > 1:
                decode_t = (m.tokens_generated - 1) / m.tokens_per_sec
            else:
                decode_t = 0.0
            lat.append(m.time_to_first_tok_s + decode_t)

    total_inference_time    = sum(lat)
    result.inference_time_s = total_inference_time

    # Aggregate throughput (all questions)
    result.avg_tokens_per_sec = (
        total_tokens / total_inference_time if total_inference_time > 0 else 0.0
    )

    # Clean throughput (non-truncated only)
    clean_pairs = [
        (metrics_list[i].tokens_generated, lat[i])
        for i, qr in enumerate(question_results)
        if not qr.truncated and i < len(metrics_list)
    ]
    if clean_pairs:
        clean_tokens = sum(t for t, _ in clean_pairs)
        clean_time   = sum(l for _, l in clean_pairs)
        result.avg_tokens_per_sec_clean = (
            clean_tokens / clean_time if clean_time > 0 else None
        )

    result.latency_values   = lat
    result.avg_latency_s    = statistics.mean(lat) if lat else 0.0
    result.median_latency_s = statistics.median(lat) if lat else 0.0
    result.p95_latency_s    = _percentile(lat, 95)

    # TTFT
    if ttft_values:
        result.ttft_values = ttft_values
        result.avg_ttft_s  = statistics.mean(ttft_values)
        result.min_ttft_s  = min(ttft_values)
        result.max_ttft_s  = max(ttft_values)

    # CPU
    if cpu_percents:
        result.avg_cpu_percent = statistics.mean(cpu_percents)

    # Thermal throttle warning
    if result.cpu_temp_c is not None and result.cpu_temp_c >= THROTTLE_WARN_C:
        result.thermal_throttle_warning = True

    result.success_count = len(metrics_list)
    result.fail_count    = 0

    return result


# Benchmark Execution

async def run_benchmark(
    bridge,
    questions: list[str] = BENCHMARK_QUESTIONS,
    *,
    max_tokens: Optional[int] = None,
    profile: str = "medium",
    temperature: float = 0.7,
    timeout_s: float = 30.0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[BenchmarkResults, list[InferenceMetrics]]:
    """
    Run the benchmark suite off the event loop.

    Per-question token budgets
    When max_tokens is None (the default), each question uses its entry from
    QUESTION_MAX_TOKENS if available, falling back to the profile default.
    This prevents short-answer questions (arithmetic, capitals) from being
    padded to 128–256 tokens, which would inflate latency and make tok/s
    variance look artificially flat.

    Passing an explicit max_tokens overrides all per-question budgets.

    Truncation detection
    After each question, tokens_generated is compared to the budget used.
    Truncated questions are flagged in QuestionResult.truncated and counted
    in BenchmarkResults.truncated_count.  avg_tokens_per_sec_clean excludes
    them so callers can see clean decode throughput separately.
    """
    profile_default = BENCHMARK_PROFILES.get(profile, 128)

    metrics_list: list[InferenceMetrics]   = []
    question_results: list[QuestionResult] = []
    latency_values: list[float]            = []
    success_count = 0
    fail_count    = 0

    load_time_s = getattr(bridge, "_load_time_s", 0.0)

    # Warmup (excluded from stats, but timed)
    t_warmup = time.perf_counter()
    try:
        await asyncio.to_thread(
            bridge.infer, "Hello", max_tokens=16, temperature=0.0
        )
    except Exception:
        pass
    warmup_time_s = time.perf_counter() - t_warmup

    t_suite_start = time.perf_counter()

    # Per-question inference
    for i, question in enumerate(questions):
        if on_progress:
            on_progress(i + 1, len(questions), question)

        # Resolve token budget for this specific question.
        if max_tokens is not None:
            q_max_tokens = max_tokens
        else:
            q_max_tokens = QUESTION_MAX_TOKENS.get(question, profile_default)

        t_q = time.perf_counter()
        try:
            _output, metrics = await asyncio.wait_for(
                asyncio.to_thread(
                    bridge.infer,
                    question,
                    max_tokens=q_max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout_s,
            )
            elapsed = time.perf_counter() - t_q

            truncated = _is_truncated(metrics.tokens_generated, q_max_tokens)

            qr = QuestionResult(
                index               = i,
                question            = question,
                tokens_generated    = metrics.tokens_generated,
                tokens_per_sec      = metrics.tokens_per_sec,
                time_to_first_tok_s = metrics.time_to_first_tok_s,
                latency_s           = elapsed,
                max_tokens_used     = q_max_tokens,
                truncated           = truncated,
            )

            metrics_list.append(metrics)
            question_results.append(qr)
            latency_values.append(elapsed)
            success_count += 1

        except asyncio.TimeoutError:
            fail_count += 1
            question_results.append(QuestionResult(
                index=i, question=question, tokens_generated=0,
                tokens_per_sec=0.0, time_to_first_tok_s=0.0,
                latency_s=timeout_s, max_tokens_used=q_max_tokens,
                truncated=False, timed_out=True,
            ))
        except Exception:
            fail_count += 1
            question_results.append(QuestionResult(
                index=i, question=question, tokens_generated=0,
                tokens_per_sec=0.0, time_to_first_tok_s=0.0,
                latency_s=0.0, max_tokens_used=q_max_tokens,
                truncated=False, error=True,
            ))

    total_time = time.perf_counter() - t_suite_start

    # Aggregate
    if metrics_list:
        results = aggregate_metrics(
            metrics_list, question_results, load_time_s, latency_values
        )
    else:
        results = BenchmarkResults(load_time_s=load_time_s)

    results.total_time_s   = total_time
    results.warmup_time_s  = warmup_time_s
    results.success_count  = success_count
    results.fail_count     = fail_count
    results.question_count = len(questions)

    # Perplexity (single pass after all inference)
    if getattr(bridge, "compute_perplexity", False):
        perplexity_text = "\n".join(questions)
        try:
            ppl = await asyncio.to_thread(bridge.perplexity, perplexity_text)
            results.perplexity = ppl
        except Exception:
            results.perplexity = None

    return results, metrics_list