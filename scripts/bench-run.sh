#!/usr/bin/env bash
# Runs on the benchmark machine, started by bench-dedibox.sh or bench-nixlab.sh. Builds the
# binom_variants benchmark for the x86-64 baseline that the wheels target and runs it with one
# thread pinned to a core and with every thread; --full also builds and runs it for the host CPU
# (target-cpu=native). Results go to results/: Criterion's output per configuration, and its
# estimates under results/criterion-<target>/, one baseline per run.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z ${RUSTFRC_BENCH_SHELL:-} ]]; then
	# pyo3's build script needs a Python interpreter, even for the benchmark.
	RUSTFRC_BENCH_SHELL=1 exec nix shell nixpkgs#cargo nixpkgs#rustc nixpkgs#gcc \
		nixpkgs#python3 nixpkgs#util-linux --command bash "$0" "$@"
fi

targets=(baseline)
if [[ ${1:-} == --full ]]; then
	targets=(baseline native)
	shift
fi
filter=${1:-}
# The process group, which bench-dedibox.sh stop ends.
ps -o pgid= $$ | tr -d " " > pgid
date -u +%Y%m%dT%H%M%SZ > run-id
touch running
trap 'rm -f running; touch done' EXIT
rm -rf results
mkdir -p results

{
	echo "run: $(cat run-id), revision $(cat revision), targets ${targets[*]}, filter '${filter}'"
	uname -a
	rustc -vV
	cargo -V
	echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo none)"
	echo "smt: $(cat /sys/devices/system/cpu/smt/active 2>/dev/null || echo unknown)"
	uptime
	lscpu
} > env.txt

# The samplers must be right on this CPU before their speed means anything: the tests compare
# every path that the CPU has, such as AVX-512, with the portable one.
echo "=== tests ($(date -u +%H:%M:%S))"
cargo test --release --lib binom_variants 2>&1 | tee results/tests.txt

for target in "${targets[@]}"; do
	if [[ $target == native ]]; then rustflags="-C target-cpu=native"; else rustflags=""; fi
	# Saved estimates of earlier runs would otherwise show up in this run's results.
	rm -rf "target/$target/criterion"
	for threads in 1 all; do
		name=$target-t$threads
		echo "=== $name ($(date -u +%H:%M:%S))"
		if [[ $threads == 1 ]]; then
			# Core 2, away from core 0, which takes most interrupts.
			pin=(taskset -c 2)
			export RAYON_NUM_THREADS=1
		else
			pin=()
			unset RAYON_NUM_THREADS
		fi
		# Build on every core first, so that only the benchmark is pinned.
		RUSTFLAGS=$rustflags CARGO_TARGET_DIR=target/$target cargo bench --bench binom_variants --no-run
		RUSTFLAGS=$rustflags CARGO_TARGET_DIR=target/$target "${pin[@]}" \
			cargo bench --bench binom_variants -- --noplot --save-baseline "$name" ${filter:+"$filter"} \
			2>&1 | tee "results/$name.txt"
	done
	cp -r "target/$target/criterion" "results/criterion-$target"
done
echo "=== finished ($(date -u +%H:%M:%S))"
