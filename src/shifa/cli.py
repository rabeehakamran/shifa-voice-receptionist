"""Text-mode phone call with Hina. Run: python -m shifa.cli [--phone 03001234567]"""

import argparse

from rich.console import Console

from shifa import llm
from shifa.session import CallSession

console = Console()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phone", default="03001234567")
    args = p.parse_args()
    llm.warmup()  # opens connections while the caller reads the greeting
    call = CallSession(args.phone)
    console.print(f"[dim]Call {call.call_id} from {args.phone}. Type 'bye' to hang up.[/dim]")
    console.print("[bold magenta]Hina:[/] Assalam o Alaikum, Shifa Family Clinic. Main Hina, kaise madad kar sakti hoon?")
    while True:
        text = console.input("[bold cyan]You:[/] ").strip()
        if not text:
            continue
        if text.lower() in {"bye", "quit", "exit", "allah hafiz"}:
            break
        r = call.turn(text)
        console.print(f"[bold magenta]Hina:[/] {r.reply}")
        t = " ".join(f"{k}={v}ms" for k, v in r.timings_ms.items())
        cites = f" cites={r.citations}" if r.citations else ""
        console.print(f"[dim]  route={r.route} risk={r.risk}{cites} | {t}[/dim]")
    summary = call.end()
    console.print(f"[dim]Call ended. outcome={summary.outcome} sentiment={summary.sentiment}\n"
                  f"summary: {summary.summary}[/dim]")


if __name__ == "__main__":
    main()
