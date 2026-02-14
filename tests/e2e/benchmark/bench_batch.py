#!/Users/hw/mlx-lm-server/.venv/bin/python
"""Batch inference throughput benchmark: measure throughput scaling with batch size.

Sends k simultaneous streaming requests (k=1..8) using distinct long prompts
(~500-1000 tokens each) and measures whether aggregate throughput increases
with batch size.  Each request uses a different prompt to avoid KV cache reuse.

Usage:
    python bench_batch.py [--server-url http://localhost:8080] [--max-tokens 128] [--runs 2]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

RESULTS_DIR = Path("/tmp/kimi-bench-results")

# ---------------------------------------------------------------------------
# Prompts — 8 distinct news-article-length passages (~500-1000 tokens each)
# ---------------------------------------------------------------------------

PROMPTS = [
    # 1 — AI Regulation
    (
        "The following is a detailed news article about artificial intelligence regulation.\n\n"
        "Governments around the world are grappling with how to regulate artificial intelligence "
        "as the technology rapidly advances and becomes embedded in critical sectors. The European "
        "Union's AI Act, which came into force in 2024, represents the most comprehensive "
        "legislative framework to date. It classifies AI systems by risk level — from minimal "
        "risk applications like spam filters to high-risk uses in healthcare, law enforcement, "
        "and critical infrastructure. Developers of high-risk systems must conduct conformity "
        "assessments, maintain technical documentation, and ensure human oversight.\n\n"
        "In the United States, the approach has been more fragmented. Executive orders have "
        "directed federal agencies to develop sector-specific guidelines, while states like "
        "California and New York have introduced their own bills targeting algorithmic bias, "
        "deepfakes, and automated decision-making in employment. Industry groups argue that "
        "overly prescriptive regulation could stifle innovation, while civil rights organizations "
        "insist that guardrails are essential to prevent discriminatory outcomes.\n\n"
        "China has taken a different path, implementing regulations that focus on generative AI "
        "content, algorithmic recommendation systems, and deep synthesis technology. These rules "
        "require service providers to conduct security assessments and register algorithms with "
        "government authorities. The geopolitical dimension adds another layer of complexity, as "
        "nations compete for AI supremacy while trying to establish international norms.\n\n"
        "Meanwhile, international bodies like the OECD and the G7 have attempted to establish "
        "common principles — transparency, accountability, fairness, and safety — but translating "
        "these high-level ideals into enforceable standards remains a formidable challenge. The "
        "pace of AI development consistently outstrips the pace of regulation, leaving policymakers "
        "in a perpetual game of catch-up.\n\n"
        "Based on the article above, summarize the key regulatory approaches and their trade-offs."
    ),

    # 2 — Climate Change
    (
        "The following is a detailed news article about climate change and global response.\n\n"
        "The latest report from the Intergovernmental Panel on Climate Change paints a stark "
        "picture: global temperatures have already risen by approximately 1.2 degrees Celsius "
        "above pre-industrial levels, and without dramatic reductions in greenhouse gas emissions, "
        "the world is on track to exceed the 1.5-degree threshold within the next decade. The "
        "consequences are already visible — more frequent and intense heatwaves, accelerating ice "
        "sheet loss in Greenland and Antarctica, rising sea levels threatening coastal communities, "
        "and shifting precipitation patterns disrupting agriculture worldwide.\n\n"
        "The transition to renewable energy is accelerating, with solar and wind now the cheapest "
        "sources of new electricity generation in most markets. Global investment in clean energy "
        "exceeded 500 billion dollars in 2024, driven by declining technology costs, supportive "
        "policies, and growing corporate commitments. Battery storage capacity is expanding "
        "rapidly, addressing the intermittency challenge that has long been cited as a limitation "
        "of renewables.\n\n"
        "However, significant obstacles remain. Fossil fuel infrastructure represents trillions "
        "of dollars in sunk costs, and powerful economic interests resist rapid phase-outs. "
        "Developing nations argue that they need continued access to affordable energy to lift "
        "their populations out of poverty, and that wealthy nations bear the greatest historical "
        "responsibility for emissions. Climate finance commitments from developed countries have "
        "consistently fallen short of pledges.\n\n"
        "Adaptation is becoming an increasingly urgent priority alongside mitigation. Cities are "
        "investing in flood defenses, heat-resilient infrastructure, and early warning systems. "
        "Agricultural researchers are developing drought-resistant crop varieties. But adaptation "
        "has limits, and some impacts — like species extinction and permafrost thaw releasing "
        "stored methane — may be irreversible once tipping points are crossed.\n\n"
        "Based on the article above, analyze the main challenges and opportunities in addressing "
        "climate change."
    ),

    # 3 — Space Exploration
    (
        "The following is a detailed news article about the future of space exploration.\n\n"
        "Space exploration is entering a new golden age, driven by a combination of government "
        "programs and private sector innovation. NASA's Artemis program aims to return humans to "
        "the Moon and establish a sustained presence through the Lunar Gateway station. SpaceX's "
        "Starship, the largest and most powerful rocket ever built, promises to dramatically "
        "reduce launch costs and enable missions to Mars. Blue Origin, Rocket Lab, and a growing "
        "constellation of smaller companies are expanding access to orbit.\n\n"
        "The commercial space economy is booming. Satellite internet constellations like Starlink "
        "are providing broadband connectivity to remote areas. Earth observation satellites support "
        "agriculture, disaster response, and climate monitoring. Space tourism, once the domain of "
        "science fiction, has become a reality with multiple companies offering suborbital and "
        "orbital flights. The global space economy is projected to exceed one trillion dollars "
        "by 2030.\n\n"
        "International collaboration and competition shape the landscape in complex ways. China's "
        "space program has achieved remarkable milestones, including operating its own space "
        "station, landing rovers on the far side of the Moon, and returning lunar samples. India "
        "successfully landed near the lunar south pole with Chandrayaan-3. Japan and the European "
        "Space Agency contribute critical capabilities in robotics, science instruments, and "
        "launch services.\n\n"
        "The scientific discoveries continue to inspire. The James Webb Space Telescope has "
        "revealed the earliest galaxies, characterized exoplanet atmospheres, and provided new "
        "insights into star formation. Mars rovers and orbiters are building an increasingly "
        "detailed understanding of the planet's geology and potential for past life. Sample return "
        "missions promise to revolutionize our understanding of the solar system.\n\n"
        "Based on the article above, discuss the key drivers and implications of the new space era."
    ),

    # 4 — Quantum Computing
    (
        "The following is a detailed news article about quantum computing progress.\n\n"
        "Quantum computing has moved from theoretical promise to practical milestone. In recent "
        "years, multiple organizations have demonstrated quantum processors with hundreds of "
        "qubits, and roadmaps call for machines with thousands of logical qubits by the end of "
        "the decade. Google's quantum supremacy demonstration showed that a quantum processor "
        "could solve a specific problem exponentially faster than classical supercomputers, while "
        "IBM has steadily scaled its quantum hardware and software ecosystem.\n\n"
        "The potential applications are transformative. In drug discovery, quantum simulations "
        "could model molecular interactions with unprecedented accuracy, dramatically accelerating "
        "the identification of new therapeutic compounds. In materials science, quantum computers "
        "could design novel materials with specific properties — better superconductors, more "
        "efficient solar cells, stronger lightweight composites. Financial institutions are "
        "exploring quantum algorithms for portfolio optimization, risk analysis, and fraud "
        "detection.\n\n"
        "However, formidable technical challenges remain. Current quantum processors are noisy "
        "and error-prone, requiring sophisticated error correction techniques that consume most "
        "available qubits for overhead rather than computation. Maintaining quantum coherence "
        "demands extreme conditions — temperatures near absolute zero for superconducting qubits, "
        "precise laser control for trapped ions. The engineering required to scale these systems "
        "while maintaining fidelity is enormous.\n\n"
        "The cybersecurity implications are equally significant. Quantum computers powerful enough "
        "to break current encryption standards could render much of today's digital security "
        "infrastructure obsolete. This has spurred a global effort to develop and deploy "
        "post-quantum cryptography standards. NIST has already selected several candidate "
        "algorithms, and organizations worldwide are beginning the complex process of migrating "
        "their systems.\n\n"
        "Based on the article above, evaluate the current state and future prospects of quantum "
        "computing."
    ),

    # 5 — Global Economy
    (
        "The following is a detailed news article about the global economic outlook.\n\n"
        "The global economy faces a complex landscape of interconnected challenges and "
        "opportunities. Inflation, which surged to multi-decade highs following the pandemic and "
        "the energy price shock, has gradually moderated as central banks implemented aggressive "
        "monetary tightening. However, the path back to target inflation rates has been uneven, "
        "with services inflation proving particularly sticky in many economies. Central banks face "
        "the delicate task of determining when to ease rates without reigniting price pressures.\n\n"
        "Geopolitical fragmentation is reshaping global trade patterns. Supply chain "
        "diversification, often described as friend-shoring or near-shoring, is accelerating as "
        "companies seek to reduce dependence on any single country. The semiconductor industry "
        "exemplifies this trend, with massive government subsidies driving new chip fabrication "
        "facilities in the United States, Europe, Japan, and India. While this may enhance "
        "resilience, it also raises costs and reduces the efficiency gains from globalization.\n\n"
        "The labor market presents its own paradoxes. Many developed economies face structural "
        "labor shortages driven by aging populations and changing worker preferences. At the same "
        "time, artificial intelligence and automation are transforming job requirements, creating "
        "demand for new skills while potentially displacing certain roles. The net effect on "
        "employment remains hotly debated among economists.\n\n"
        "Sovereign debt levels have risen sharply, constrained by higher interest rates that "
        "increase servicing costs. Fiscal space for addressing long-term challenges — climate "
        "adaptation, infrastructure modernization, social safety nets — is increasingly limited. "
        "Emerging markets face particular vulnerabilities, with dollar-denominated debt becoming "
        "more expensive as the US currency remains strong.\n\n"
        "Based on the article above, identify the key economic risks and potential growth drivers "
        "for the coming years."
    ),

    # 6 — Healthcare Innovation
    (
        "The following is a detailed news article about healthcare innovation and technology.\n\n"
        "Healthcare is undergoing a technological transformation that promises to fundamentally "
        "change how diseases are diagnosed, treated, and prevented. Artificial intelligence is at "
        "the forefront, with machine learning models demonstrating diagnostic accuracy comparable "
        "to or exceeding human specialists in radiology, pathology, and dermatology. AI-powered "
        "tools are accelerating drug discovery by predicting protein structures, identifying "
        "potential drug candidates, and optimizing clinical trial design.\n\n"
        "Genomic medicine is moving from research to routine clinical practice. The cost of whole "
        "genome sequencing has dropped below two hundred dollars, making it feasible for broader "
        "population screening. Pharmacogenomics — tailoring drug selection and dosing based on "
        "individual genetic profiles — is reducing adverse drug reactions and improving treatment "
        "efficacy. Gene therapies and gene editing technologies like CRISPR are offering "
        "potential cures for previously intractable genetic diseases.\n\n"
        "Digital health technologies are expanding access to care, particularly in underserved "
        "areas. Telemedicine adoption, which surged during the pandemic, has remained elevated as "
        "patients and providers recognize its convenience. Wearable devices continuously monitor "
        "vital signs, detect arrhythmias, and track chronic conditions. Remote patient monitoring "
        "programs are reducing hospital readmissions and enabling earlier intervention.\n\n"
        "However, significant challenges accompany these advances. Health data privacy and "
        "security concerns are intensifying as more sensitive information is digitized and shared. "
        "Regulatory frameworks struggle to keep pace with rapidly evolving technologies. Health "
        "equity remains a critical issue, as innovative treatments are often prohibitively "
        "expensive and access varies dramatically between and within countries. The integration "
        "of AI into clinical workflows requires careful validation, transparent algorithms, and "
        "physician trust.\n\n"
        "Based on the article above, assess the most promising healthcare innovations and their "
        "barriers to adoption."
    ),

    # 7 — Renewable Energy
    (
        "The following is a detailed news article about the renewable energy transition.\n\n"
        "The global energy landscape is undergoing its most dramatic transformation since the "
        "industrial revolution. Renewable energy sources — primarily solar, wind, and "
        "hydroelectric — now account for over thirty percent of global electricity generation, "
        "a share that continues to grow as costs decline and deployment accelerates. Solar "
        "photovoltaic capacity additions have exceeded all other sources combined for several "
        "consecutive years, driven by manufacturing scale in China and supportive policies "
        "worldwide.\n\n"
        "Energy storage is emerging as the critical enabler of a renewable-dominated grid. "
        "Lithium-ion battery costs have fallen by over ninety percent in the past decade, making "
        "grid-scale storage economically viable. New chemistries — sodium-ion, iron-air, and "
        "solid-state batteries — promise further cost reductions and reduced reliance on scarce "
        "materials like lithium and cobalt. Long-duration storage technologies, including "
        "compressed air, pumped hydro, and green hydrogen, are essential for managing seasonal "
        "variability.\n\n"
        "The electrification of transportation is accelerating the energy transition. Electric "
        "vehicle sales have grown exponentially, with several major markets setting phase-out "
        "dates for internal combustion engine vehicles. This is driving massive investment in "
        "charging infrastructure, grid upgrades, and battery recycling capabilities. The "
        "intersection of EVs and smart grids creates opportunities for vehicle-to-grid services "
        "that can help balance supply and demand.\n\n"
        "Grid modernization presents both technical and political challenges. Transmission "
        "infrastructure built for centralized fossil fuel plants must be redesigned for "
        "distributed renewable generation. Permitting and siting of new transmission lines and "
        "renewable installations face opposition from local communities concerned about visual "
        "impact, land use, and property values. Workforce transition — retraining fossil fuel "
        "workers for clean energy jobs — requires sustained investment and community support.\n\n"
        "Based on the article above, evaluate the current state of the energy transition and "
        "remaining barriers."
    ),

    # 8 — Cybersecurity
    (
        "The following is a detailed news article about the evolving cybersecurity landscape.\n\n"
        "Cybersecurity threats are growing in sophistication, scale, and impact as digital "
        "transformation accelerates across every sector. Ransomware attacks have evolved from "
        "opportunistic malware distribution to highly organized criminal enterprises that "
        "specifically target high-value organizations — hospitals, critical infrastructure "
        "operators, municipal governments, and large corporations. The average ransom payment "
        "has increased dramatically, and the total economic cost including downtime, recovery, "
        "and reputational damage reaches into the billions annually.\n\n"
        "Nation-state cyber operations represent an equally serious threat. Advanced persistent "
        "threat groups, often backed by government resources, conduct espionage, intellectual "
        "property theft, and pre-positioning for potential conflict. Supply chain attacks, in "
        "which adversaries compromise widely used software or hardware to gain access to "
        "thousands of downstream targets, have proven particularly devastating. The SolarWinds "
        "and Log4j incidents demonstrated the cascading risks inherent in modern software "
        "ecosystems.\n\n"
        "Artificial intelligence is transforming both offense and defense in cybersecurity. "
        "Attackers use AI to craft more convincing phishing messages, automate vulnerability "
        "discovery, and evade detection systems. Defenders leverage AI for anomaly detection, "
        "threat intelligence analysis, and automated incident response. The arms race between "
        "AI-powered attack and defense capabilities is intensifying, with neither side holding "
        "a decisive advantage.\n\n"
        "Organizations are increasingly adopting zero-trust architectures that assume no user or "
        "device should be inherently trusted, regardless of whether they are inside or outside "
        "the network perimeter. Multi-factor authentication, micro-segmentation, continuous "
        "monitoring, and least-privilege access controls form the foundation of this approach. "
        "However, implementing zero trust across complex legacy environments is technically "
        "challenging and resource-intensive.\n\n"
        "Based on the article above, analyze the most significant cybersecurity threats and "
        "defense strategies."
    ),
]

BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_model_name(server_url: str) -> str:
    """Fetch the loaded model name from the server."""
    resp = httpx.get(f"{server_url}/v1/models", timeout=10.0)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models loaded on server")
    return models[0]["id"]


def get_health(url: str) -> dict | None:
    """Fetch server health stats."""
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
            resp = client.get(f"{url}/health", timeout=10.0)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def parse_sse_stream(response: httpx.Response):
    """Yield parsed SSE data events from a streaming response."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Single streaming request (one thread)
# ---------------------------------------------------------------------------

def single_streaming_request(
    url: str, prompt: str, max_tokens: int, request_idx: int, model: str = "default"
) -> dict:
    """Execute a single streaming chat request and measure prefill / decode."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft = None
    token_count = 0
    finish_reason = None
    usage_info = {}
    error = None

    try:
        with httpx.Client(timeout=httpx.Timeout(600.0)) as client:
            with client.stream(
                "POST", f"{url}/v1/chat/completions", json=payload, timeout=600.0
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    return {
                        "request_idx": request_idx,
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "total_time_s": round(time.perf_counter() - t0, 4),
                    }

                for chunk in parse_sse_stream(resp):
                    now = time.perf_counter()

                    if "usage" in chunk and not chunk.get("choices"):
                        usage_info = chunk["usage"]
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    fr = choices[0].get("finish_reason")

                    if content:
                        if ttft is None:
                            ttft = now - t0
                        token_count += 1

                    if fr is not None:
                        finish_reason = fr

    except Exception as e:
        error = str(e)

    total_time = time.perf_counter() - t0
    prompt_tokens = usage_info.get("prompt_tokens", 0)
    completion_tokens = usage_info.get("completion_tokens", token_count)

    if error:
        return {
            "request_idx": request_idx,
            "error": error,
            "total_time_s": round(total_time, 4),
        }

    # Derive timing metrics
    prefill_time_s = ttft if ttft is not None else total_time
    decode_time_s = total_time - prefill_time_s if ttft is not None else 0.0
    prefill_tok_s = (
        round(prompt_tokens / prefill_time_s, 1) if prefill_time_s > 0 and prompt_tokens > 0 else 0.0
    )
    decode_tok_s = (
        round(completion_tokens / decode_time_s, 2) if decode_time_s > 0 and completion_tokens > 0 else 0.0
    )
    throughput_tok_s = (
        round((prompt_tokens + completion_tokens) / total_time, 2) if total_time > 0 else 0.0
    )

    return {
        "request_idx": request_idx,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_time_s": round(prefill_time_s, 4),
        "prefill_tok_s": prefill_tok_s,
        "decode_time_s": round(decode_time_s, 4),
        "decode_tok_s": decode_tok_s,
        "throughput_tok_s": throughput_tok_s,
        "total_time_s": round(total_time, 4),
        "finish_reason": finish_reason,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Run one batch (k simultaneous requests)
# ---------------------------------------------------------------------------

def run_batch(
    url: str, batch_size: int, max_tokens: int, model: str, run_idx: int
) -> dict:
    """Send *batch_size* simultaneous streaming requests, measure wall-clock."""
    # Each request gets a distinct prompt (no cache benefit)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(batch_size)]

    wall_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(single_streaming_request, url, p, max_tokens, i, model): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_clock_s = time.perf_counter() - wall_start
    results.sort(key=lambda r: r["request_idx"])

    # Aggregate
    successful = [r for r in results if not r.get("error")]
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in successful)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    total_tokens = total_prompt_tokens + total_completion_tokens

    aggregate_throughput_tok_s = round(total_tokens / wall_clock_s, 2) if wall_clock_s > 0 else 0.0

    return {
        "run_idx": run_idx,
        "wall_clock_s": round(wall_clock_s, 4),
        "aggregate_throughput_tok_s": aggregate_throughput_tok_s,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "requests": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch inference throughput benchmark")
    parser.add_argument("--server-url", default=None,
                        help="Server URL (default: $SERVER_URL or http://localhost:8080)")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of runs per batch size (default: 2)")
    parser.add_argument("--batch-sizes", type=str, default=None,
                        help="Comma-separated batch sizes (default: 1,2,3,4,5,6,7,8)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    url = (args.server_url or os.environ.get("SERVER_URL", "http://localhost:8080")).rstrip("/")
    max_tokens = args.max_tokens
    num_runs = args.runs
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes
        else BATCH_SIZES
    )

    # --- Health check ---
    health = get_health(url)
    if health is None:
        print(f"ERROR: Server not reachable at {url}", file=sys.stderr)
        sys.exit(1)
    print(f"Server health: {health.get('status', 'unknown')}")

    model_name = get_model_name(url)
    print(f"Model: {model_name}")

    # Compute average prompt token count (estimated from first prompt via a
    # quick non-timing request is impractical; we will fill it from actual runs).
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch size: {num_runs}")
    print(f"Max tokens: {max_tokens}")
    print()

    # --- Run benchmarks ---
    all_batch_results = []  # one entry per batch_size
    k1_avg_throughput = None
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    def _save_partial():
        pf = RESULTS_DIR / f"bench_batch_{ts}_partial.json"
        pf.write_text(json.dumps(
            {"benchmark": "bench_batch", "results": all_batch_results, "partial": True},
            indent=2, default=str,
        ))

    try:
        for k in batch_sizes:
            print(f"=== Batch size k={k} ===")
            runs = []
            for run_idx in range(num_runs):
                print(f"  Run {run_idx + 1}/{num_runs} ... ", end="", flush=True)
                run_result = run_batch(url, k, max_tokens, model_name, run_idx)

                ok = run_result["successful"]
                errs = run_result["errors"]
                print(
                    f"wall={run_result['wall_clock_s']:.2f}s, "
                    f"throughput={run_result['aggregate_throughput_tok_s']:.1f} tok/s, "
                    f"{ok}/{k} ok" + (f", {errs} errors" if errs else "")
                )
                runs.append(run_result)

                # Brief pause between runs to let the server settle
                if run_idx < num_runs - 1:
                    time.sleep(1)

            # Aggregate across runs
            throughputs = [r["aggregate_throughput_tok_s"] for r in runs]
            avg_throughput = round(statistics.mean(throughputs), 2)

            # Per-request latencies across all runs
            all_latencies = []
            for r in runs:
                for req in r["requests"]:
                    if not req.get("error"):
                        all_latencies.append(req["total_time_s"])

            avg_latency = round(statistics.mean(all_latencies), 4) if all_latencies else 0.0
            sorted_latencies = sorted(all_latencies)
            p95_idx = max(0, math.ceil(len(sorted_latencies) * 0.95) - 1)
            p95_latency = round(sorted_latencies[p95_idx], 4) if sorted_latencies else 0.0

            # Prompt tokens average (from actual requests)
            all_prompt_tokens = []
            for r in runs:
                for req in r["requests"]:
                    pt = req.get("prompt_tokens", 0)
                    if pt > 0:
                        all_prompt_tokens.append(pt)
            avg_prompt_tokens = round(statistics.mean(all_prompt_tokens)) if all_prompt_tokens else 0

            if k1_avg_throughput is None:
                k1_avg_throughput = avg_throughput
            speedup = round(avg_throughput / k1_avg_throughput, 2) if k1_avg_throughput > 0 else 0.0

            batch_entry = {
                "batch_size": k,
                "runs": runs,
                "avg_aggregate_throughput_tok_s": avg_throughput,
                "avg_per_request_latency_s": avg_latency,
                "p95_per_request_latency_s": p95_latency,
                "avg_prompt_tokens": avg_prompt_tokens,
                "speedup_vs_k1": speedup,
            }
            all_batch_results.append(batch_entry)
            _save_partial()

            # Pause between batch sizes
            if k != batch_sizes[-1]:
                time.sleep(2)
    except Exception as e:
        print(f"\nFATAL: {e}", file=sys.stderr)
        _save_partial()
        print(f"Partial results ({len(all_batch_results)} batches) saved", file=sys.stderr)
        raise

    # --- Save results ---
    health_after = get_health(url)

    # Overall average prompt tokens
    all_pt = []
    for br in all_batch_results:
        if br["avg_prompt_tokens"] > 0:
            all_pt.append(br["avg_prompt_tokens"])
    prompt_tokens_avg = round(statistics.mean(all_pt)) if all_pt else 0

    output = {
        "benchmark": "batch_inference",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_url": url,
        "model": model_name,
        "max_tokens": max_tokens,
        "runs_per_batch_size": num_runs,
        "prompt_tokens_avg": prompt_tokens_avg,
        "health_before": health,
        "health_after": health_after,
        "results": all_batch_results,
    }
    outfile = RESULTS_DIR / f"bench_batch_{ts}.json"
    outfile.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {outfile}")

    partial_file = RESULTS_DIR / f"bench_batch_{ts}_partial.json"
    if partial_file.exists():
        partial_file.unlink()

    # --- Console summary table ---
    print()
    print("Batch Inference Throughput Benchmark")
    print(f"Model: {model_name}")
    print(f"Prompts: ~{prompt_tokens_avg} tokens each, max_tokens={max_tokens}")
    print()
    print(
        f"{'k':>4} | {'Throughput (tok/s)':>18} | {'Avg Latency (s)':>16} | "
        f"{'P95 Latency (s)':>16} | {'Speedup vs k=1':>15}"
    )
    print("-" * 80)
    for br in all_batch_results:
        print(
            f"{br['batch_size']:>4} | "
            f"{br['avg_aggregate_throughput_tok_s']:>18.1f} | "
            f"{br['avg_per_request_latency_s']:>16.2f} | "
            f"{br['p95_per_request_latency_s']:>16.2f} | "
            f"{br['speedup_vs_k1']:>14.2f}x"
        )

    print()


if __name__ == "__main__":
    main()
