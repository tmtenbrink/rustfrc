#!/usr/bin/env bash
# Runs the binom_variants benchmark on borink-dedibox, dedicated hardware for benchmarks
# (infra/scaleway-dedibox in mono). Needs an SSH session from mono's
# infra/hetzner-node/scripts/grant-session.sh.
#
#   scripts/bench-dedibox.sh start [--full] [FILTER]   copy this checkout over and start a run
#   scripts/bench-dedibox.sh status [LINES]            show the end of the run's log
#   scripts/bench-dedibox.sh stop                      end the run
#   scripts/bench-dedibox.sh fetch                     copy the results to bench-results/dedibox/<run>
#
# A run builds for the x86-64 baseline of the wheels and takes about 5 minutes; --full adds a build
# for the host CPU. FILTER is a Criterion filter, a regular expression over benchmark names such
# as 'binom_2000'. The run itself is scripts/bench-run.sh; it leaves a file `done` when it ends.
set -euo pipefail
cd "$(dirname "$0")/.."

mono=${MONO:-$HOME/files/c/me/mono}
ssh_dir=$mono/infra/scaleway-dedibox/ssh
remote=rustfrc-bench

on_dedibox() {
	ssh -F "$ssh_dir/config" -o UserKnownHostsFile="$ssh_dir/known_hosts" borink-dedibox "$@"
}

case "${1:-}" in
start)
	shift
	if on_dedibox "test -e $remote/running"; then
		echo "a run is still going on borink-dedibox; see: $0 status" >&2
		exit 1
	fi
	revision=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo -dirty)
	# The remote target/ stays, so a rebuild after a small change is quick.
	tar -c --exclude=./target --exclude=./.venv --exclude=./.git --exclude=./bench-results . |
		on_dedibox "mkdir -p $remote && cd $remote && rm -rf src benches scripts && tar -x"
	# The run goes in a subshell of its own, so that nothing keeps the SSH channel open.
	on_dedibox "cd $remote && echo $revision > revision && rm -f done &&
		(setsid nohup bash scripts/bench-run.sh $(printf "%q " "$@") > run.log 2>&1 < /dev/null &) &&
		echo started $revision"
	;;
status)
	on_dedibox "cd $remote && tail -n ${2:-20} run.log; for f in running done; do test -e \$f && echo [\$f]; done; true"
	;;
stop)
	on_dedibox "cd $remote && test -e running && kill -TERM -- -\$(cat pgid) && echo stopped || echo 'no run is going'"
	;;
fetch)
	run=$(on_dedibox "cat $remote/run-id")
	out=bench-results/dedibox/$run
	mkdir -p "$out"
	on_dedibox "cd $remote && tar -c run.log env.txt revision results" | tar -x -C "$out"
	echo "$out"
	;;
*)
	sed -n '2,14p' "$0" >&2
	exit 2
	;;
esac
