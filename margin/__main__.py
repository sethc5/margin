"""
CLI entry point: python -m margin

Commands:
    python -m margin monitor --config margin.yaml
    python -m margin replay --config margin.yaml --data metrics.csv
    python -m margin status --config margin.yaml --data metrics.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def cmd_monitor(args):
    """Stream measurements from stdin, print health/drift/anomaly."""
    from .config import load_config
    from .streaming import Monitor

    cfg = load_config(args.config)
    parser = cfg["parser"]
    monitor = Monitor(parser, window=args.window)

    print(f"margin monitor: reading from stdin ({len(parser.baselines)} components)", file=sys.stderr)
    print(f"  components: {', '.join(sorted(parser.baselines.keys()))}", file=sys.stderr)
    print(f"  window: {args.window}", file=sys.stderr)
    print(file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError:
            # Try CSV: component=value,component=value
            values = {}
            for part in line.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        values[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            if not values:
                continue

        expr = monitor.update(values, now=datetime.now())

        # Print health
        parts = [expr.to_string()]

        # Drift annotations
        for name in sorted(parser.baselines.keys()):
            dc = monitor.drift(name)
            if dc and dc.state.value != "STABLE":
                parts.append(f"  drift({name}): {dc.state.value}({dc.direction.value})")

        # Anomaly annotations
        for name in sorted(parser.baselines.keys()):
            ac = monitor.anomaly(name)
            if ac and ac.state.value != "EXPECTED":
                parts.append(f"  anomaly({name}): {ac.state.value}")

        print("\n".join(parts))
        sys.stdout.flush()


def cmd_replay(args):
    """Replay a CSV through Monitor and print analysis."""
    from .config import load_config
    from .persist import replay_csv

    cfg = load_config(args.config)
    parser = cfg["parser"]

    monitor, snapshots = replay_csv(
        parser, args.data,
        timestamp_column=args.timestamp_column,
        window=args.window,
    )

    # Summary
    print(f"Replayed {len(snapshots)} steps across {len(parser.baselines)} components")
    print()

    # Final state
    if monitor.expression:
        print(f"Final health: {monitor.expression.to_string()}")

    # Drift summary
    print()
    print("Drift:")
    for name in sorted(parser.baselines.keys()):
        dc = monitor.drift(name)
        if dc:
            print(f"  {name}: {dc.state.value}({dc.direction.value}), rate={dc.rate:.4g}/s")
        else:
            print(f"  {name}: insufficient data")

    # Anomaly summary
    print()
    print("Anomaly:")
    for name in sorted(parser.baselines.keys()):
        ac = monitor.anomaly(name)
        if ac:
            print(f"  {name}: {ac.state.value} (z={ac.z_score:.2f})")
        else:
            print(f"  {name}: insufficient reference")

    # Correlation summary
    if monitor.correlations and monitor.correlations.correlations:
        print()
        print("Correlations:")
        for c in monitor.correlations.strongest(5):
            print(f"  {c.component_a} ~ {c.component_b}: r={c.coefficient:.3f}")

    # Write full output if requested
    if args.output:
        Path(args.output).write_text(json.dumps(snapshots, indent=2))
        print(f"\nFull snapshots written to {args.output}")


def cmd_status(args):
    """One-shot: parse values from args and print health."""
    from .config import load_config

    cfg = load_config(args.config)
    parser = cfg["parser"]

    # Parse values from remaining args: cpu=80 mem=60
    values = {}
    for part in args.values:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                values[k.strip()] = float(v.strip())
            except ValueError:
                print(f"Warning: skipping {part}", file=sys.stderr)

    if not values:
        print("Usage: python -m margin status --config margin.yaml cpu=80 mem=60", file=sys.stderr)
        sys.exit(1)

    expr = parser.parse(values)
    print(expr.to_string())


def main():
    ap = argparse.ArgumentParser(
        prog="python -m margin",
        description="margin CLI — typed health classification",
    )
    sub = ap.add_subparsers(dest="command")

    # monitor
    p_mon = sub.add_parser("monitor", help="Stream stdin, print health/drift/anomaly")
    p_mon.add_argument("--config", required=True, help="Path to margin config (YAML/JSON)")
    p_mon.add_argument("--window", type=int, default=100, help="Tracker window size")

    # replay
    p_rep = sub.add_parser("replay", help="Replay CSV through Monitor")
    p_rep.add_argument("--config", required=True, help="Path to margin config (YAML/JSON)")
    p_rep.add_argument("--data", required=True, help="Path to CSV data file")
    p_rep.add_argument("--timestamp-column", default=None, help="CSV column with timestamps")
    p_rep.add_argument("--window", type=int, default=100, help="Tracker window size")
    p_rep.add_argument("--output", default=None, help="Write full snapshots to JSON file")

    # status
    p_stat = sub.add_parser("status", help="One-shot health classification")
    p_stat.add_argument("--config", required=True, help="Path to margin config (YAML/JSON)")
    p_stat.add_argument("values", nargs="*", help="component=value pairs")

    args = ap.parse_args()

    if args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
