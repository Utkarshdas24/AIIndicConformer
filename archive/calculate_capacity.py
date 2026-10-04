import os
import sys
import json
import argparse

def calculate_capacity(measured_speed_x=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "results", "benchmark_summary.json")

    if measured_speed_x is None:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                measured_speed_x = data.get("overall_speed_x", 0.0)
                gpu_name = data.get("gpu", "NVIDIA RTX 3050 Laptop GPU")
        else:
            print("[!] Warning: benchmark_summary.json not found. Using default speed estimate of 10.0x.")
            measured_speed_x = 10.0
            gpu_name = "NVIDIA GeForce RTX 3050 Laptop GPU"
    else:
        gpu_name = "NVIDIA GeForce RTX 3050 Laptop GPU"

    if measured_speed_x <= 0:
        print("[!] Invalid measured speed_x. Must be > 0.")
        sys.exit(1)

    print("=" * 70)
    print("     PRODUCTION CAPACITY CALCULATION (PHASE 8)")
    print("=" * 70)
    print(f"GPU Tested              : {gpu_name}")
    print(f"Measured Speed Factor   : {measured_speed_x:.2f}x real-time")
    print("-" * 70)

    # Scenarios: 5,000 hours/month and 10,000 hours/month
    scenarios = [5000, 10000]

    for audio_hours in scenarios:
        req_compute_hours = audio_hours / measured_speed_x
        req_compute_mins = req_compute_hours * 60.0
        
        # Monthly operating hours assumption: 30 days * 24 hrs = 720 hrs total per month
        # Assuming maximum safe GPU utilization for laptop batch workload = 50% to 70% (e.g. ~12 hrs/day = 360 hrs/month)
        monthly_total_hours = 720.0
        gpu_utilization_pct = (req_compute_hours / monthly_total_hours) * 100.0

        daily_compute_hours = req_compute_hours / 30.0

        print(f"\n[+] REQUIREMENT: {audio_hours:,} Audio Hours / Month")
        print(f"    - Required GPU Compute Time : {req_compute_hours:.2f} GPU hours ({req_compute_mins:.1f} GPU minutes)")
        print(f"    - Daily Processing Time     : {daily_compute_hours:.2f} GPU hours / day")
        print(f"    - Monthly GPU Load Factor   : {gpu_utilization_pct:.1f}% of total monthly calendar time (720h)")

        if req_compute_hours <= 360:
            status = "FEASIBLE (Comfortable headroom for laptop batch processing)"
        elif req_compute_hours <= 720:
            status = "MARGINAL (Requires high GPU duty cycle on laptop hardware)"
        else:
            status = "EXCEEDED (Requires multiple GPUs or cloud worker scale-out)"

        print(f"    - Feasibility Assessment   : {status}")

    print("\n" + "=" * 70)
    print("CONCEPTS & DEFINITIONS:")
    print("=" * 70)
    print("1. Audio Hours:")
    print("   The total cumulative duration of audio call recordings (e.g., 5,000 hours).")
    print("\n2. GPU Compute Hours:")
    print("   The exact active execution time spent by the GPU executing forward passes")
    print("   and tensor computations to transcribe the audio.")
    print("\n3. Wall-Clock Processing Hours:")
    print("   The total real-world elapsed time, including file I/O, audio resampling,")
    print("   GPU memory transfers, process queuing, and idle system overhead.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate ASR production capacity")
    parser.add_argument("--speed", type=float, help="Override measured speed_x factor")
    args = parser.parse_args()
    calculate_capacity(args.speed)
