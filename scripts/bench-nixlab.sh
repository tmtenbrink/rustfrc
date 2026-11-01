#!/usr/bin/env bash
# Runs a benchmark (binom_variants by default) on the VM of mono's NixOS lab on Azure
# (infra/winlab): Standard_D8alds_v7, 8 AMD Zen 5 vCPUs, which have AVX-512 with vprolq and
# vpopcntq. Needs a lease, which only the security key grants
# (infra/winlab/scripts/grant-lease.sh nix).
#
#   scripts/bench-nixlab.sh start [--full] [--bench NAME] [FILTER]
#       copy this checkout over and start a run
#   scripts/bench-nixlab.sh status [LINES]   show the end of the run's log
#   scripts/bench-nixlab.sh stop             end the run
#   scripts/bench-nixlab.sh fetch            copy the results to bench-results/nixlab/<run>
#
# The arguments of start are those of scripts/bench-run.sh, which runs as the lab's job `bench`,
# so that it outlives the SSH session. It leaves a file `done` when it ends.
set -euo pipefail
cd "$(dirname "$0")/.."

winlab=${MONO:-$HOME/files/c/me/mono}/infra/winlab
remote=rustfrc-bench

on_vm() {
	ssh -F "$winlab/ssh/nixlab.config" -o UserKnownHostsFile="$winlab/ssh/nixlab.known_hosts" nixlab "$@"
}

case "${1:-}" in
start)
	shift
	if on_vm "test -e $remote/running"; then
		echo "a run is still going on the nixlab VM; see: $0 status" >&2
		exit 1
	fi
	revision=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo -dirty)
	tar -c --exclude=./target --exclude=./.venv --exclude=./.git --exclude=./bench-results --exclude=./gpu/target . |
		on_vm "mkdir -p $remote && cd $remote && rm -rf src benches scripts && tar -x"
	on_vm "cd $remote && echo $revision > revision && rm -f done run.log"
	"$winlab/nixlab.ts" remove bench > /dev/null
	printf 'cd ~/%s && exec bash scripts/bench-run.sh %s > run.log 2>&1\n' "$remote" "$(printf '%q ' "$@")" |
		"$winlab/nixlab.ts" start bench
	;;
status)
	on_vm "cd $remote && tail -n ${2:-20} run.log; for f in running done; do test -e \$f && echo [\$f]; done; true"
	;;
stop)
	"$winlab/nixlab.ts" remove bench
	;;
fetch)
	run=$(on_vm "cat $remote/run-id")
	out=bench-results/nixlab/$run
	mkdir -p "$out"
	on_vm "cd $remote && tar -c run.log env.txt revision results" | tar -x -C "$out"
	echo "$out"
	;;
*)
	sed -n '2,12p' "$0" >&2
	exit 2
	;;
esac
