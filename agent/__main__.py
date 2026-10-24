"""Terminal chat with the core agent.

    python -m agent                      # interactive
    python -m agent -q "question"        # one question, then exit

Needs Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""

from __future__ import annotations

import argparse
import json

from agent.agent import CoreAgent


def _show_call(name: str, args: dict) -> None:
    print(f"  [tool] {name} {json.dumps(args)[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    agent = CoreAgent(effort=args.effort)
    if args.question:
        print(agent.ask(args.question, _show_call))
        return
    print("PWR core agent (research surrogate). Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            print("\n" + agent.ask(q, _show_call))


if __name__ == "__main__":
    main()
