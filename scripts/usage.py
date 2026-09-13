#!/usr/bin/env python3
"""Report server and model utilisation recorded in logs/metrics.db.

Examples:
    python scripts/usage.py                    # last 7 days, text report
    python scripts/usage.py --last 24h
    python scripts/usage.py --last 30d --html logs/usage.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import connect, get_db_path  # noqa: E402

_DURATION = re.compile(r"^(\d+)\s*([smhdw])$", re.IGNORECASE)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_duration(text: str) -> float:
    match = _DURATION.match(text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid duration '{text}'. Use e.g. 30m, 24h, 7d, 2w.")
    return int(match.group(1)) * _UNITS[match.group(2).lower()]


def fmt_int(value) -> str:
    return f"{int(value or 0):,}"


def fmt_cost(value) -> str:
    """Unknown pricing renders as a dash, never as $0.00."""
    if value is None:
        return "—"
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def table(rows: list[tuple], headers: tuple, aligns: str) -> str:
    if not rows:
        return "  (no data)\n"
    cols = list(zip(*([headers] + [tuple(str(c) for c in r) for r in rows])))
    widths = [max(len(str(c)) for c in col) for col in cols]
    out = []

    def line(cells):
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            parts.append(str(cell).ljust(width) if align == "l" else str(cell).rjust(width))
        return "  " + "  ".join(parts)

    out.append(line(headers))
    out.append("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        out.append(line([str(c) for c in row]))
    return "\n".join(out) + "\n"


def gather(conn, since: float) -> dict:
    q = lambda sql, *a: [dict(r) for r in conn.execute(sql, a).fetchall()]  # noqa: E731

    return {
        "totals": q(
            "SELECT COUNT(*) AS calls, "
            "SUM(status='ok') AS ok, SUM(status='error') AS errors, "
            "AVG(duration_ms) AS avg_ms, MAX(duration_ms) AS max_ms "
            "FROM calls WHERE ts >= ?",
            since,
        ),
        "by_tool": q(
            "SELECT c.tool, COUNT(*) AS calls, SUM(c.status='error') AS errors, "
            "CAST(AVG(c.duration_ms) AS INT) AS avg_ms, "
            "COALESCE((SELECT SUM(m.total_tokens) FROM model_calls m WHERE m.tool = c.tool AND m.ts >= ?), 0) AS tokens "
            "FROM calls c WHERE c.ts >= ? GROUP BY c.tool ORDER BY calls DESC",
            since,
            since,
        ),
        "by_model": q(
            "SELECT model, provider, COUNT(*) AS calls, SUM(input_tokens) AS inp, "
            "SUM(output_tokens) AS outp, SUM(total_tokens) AS tokens, "
            "CASE WHEN COUNT(*) = COUNT(cost_usd) THEN SUM(cost_usd) ELSE NULL END AS cost "
            "FROM model_calls WHERE ts >= ? GROUP BY model, provider ORDER BY tokens DESC",
            since,
        ),
        "by_day": q(
            "SELECT date(ts,'unixepoch','localtime') AS day, COUNT(*) AS calls, "
            "COALESCE((SELECT SUM(m.total_tokens) FROM model_calls m "
            "  WHERE date(m.ts,'unixepoch','localtime') = date(c.ts,'unixepoch','localtime')), 0) AS tokens "
            "FROM calls c WHERE c.ts >= ? GROUP BY day ORDER BY day",
            since,
        ),
        "by_client": q(
            "SELECT COALESCE(client,'unknown') AS client, COUNT(*) AS calls "
            "FROM calls WHERE ts >= ? GROUP BY client ORDER BY calls DESC",
            since,
        ),
        "errors": q(
            "SELECT tool, COALESCE(error_type,'?') AS error_type, COUNT(*) AS n, MAX(error) AS sample "
            "FROM calls WHERE ts >= ? AND status='error' GROUP BY tool, error_type ORDER BY n DESC LIMIT 15",
            since,
        ),
        "latency": q(
            "SELECT tool, duration_ms FROM calls WHERE ts >= ? AND duration_ms IS NOT NULL ORDER BY tool, duration_ms",
            since,
        ),
    }


def percentiles(latency: list[dict]) -> dict[str, dict[str, int]]:
    by_tool: dict[str, list[int]] = {}
    for row in latency:
        by_tool.setdefault(row["tool"], []).append(row["duration_ms"])

    def pick(values: list[int], fraction: float) -> int:
        return values[min(len(values) - 1, int(len(values) * fraction))]

    out = {}
    for tool, values in by_tool.items():
        values.sort()
        out[tool] = {"p50": pick(values, 0.50), "p95": pick(values, 0.95)}
    return out


def text_report(data: dict, window: str) -> str:
    t = (data["totals"] or [{}])[0]
    pct = percentiles(data["latency"])
    total_tokens = sum(r["tokens"] or 0 for r in data["by_model"])
    costs = [r["cost"] for r in data["by_model"]]
    known = [c for c in costs if c is not None]
    cost_line = fmt_cost(sum(known)) if known else "—"
    if known and len(known) < len(costs):
        cost_line += f"  (+{len(costs) - len(known)} model(s) unpriced)"

    out = [f"\nPAL utilisation — last {window}\n" + "=" * 46 + "\n"]
    out.append(
        f"  Calls: {fmt_int(t.get('calls'))}   "
        f"OK: {fmt_int(t.get('ok'))}   Errors: {fmt_int(t.get('errors'))}\n"
        f"  Tokens: {fmt_int(total_tokens)}   Est. cost: {cost_line}\n"
        f"  Avg: {fmt_int(t.get('avg_ms'))} ms   Max: {fmt_int(t.get('max_ms'))} ms\n"
    )

    out.append("\nBy tool\n")
    out.append(
        table(
            [
                (
                    r["tool"],
                    fmt_int(r["calls"]),
                    fmt_int(r["errors"]),
                    fmt_int(r["tokens"]),
                    fmt_int(r["avg_ms"]),
                    fmt_int(pct.get(r["tool"], {}).get("p50")),
                    fmt_int(pct.get(r["tool"], {}).get("p95")),
                )
                for r in data["by_tool"]
            ],
            ("TOOL", "CALLS", "ERR", "TOKENS", "AVG ms", "p50", "p95"),
            "lrrrrrr",
        )
    )

    out.append("\nBy model\n")
    out.append(
        table(
            [
                (
                    r["model"],
                    r["provider"],
                    fmt_int(r["calls"]),
                    fmt_int(r["inp"]),
                    fmt_int(r["outp"]),
                    fmt_int(r["tokens"]),
                    fmt_cost(r["cost"]),
                )
                for r in data["by_model"]
            ],
            ("MODEL", "PROVIDER", "CALLS", "IN", "OUT", "TOTAL", "COST"),
            "llrrrrr",
        )
    )

    out.append("\nBy day\n")
    out.append(
        table(
            [(r["day"], fmt_int(r["calls"]), fmt_int(r["tokens"])) for r in data["by_day"]],
            ("DAY", "CALLS", "TOKENS"),
            "lrr",
        )
    )

    out.append("\nBy client\n")
    out.append(table([(r["client"], fmt_int(r["calls"])) for r in data["by_client"]], ("CLIENT", "CALLS"), "lr"))

    if data["errors"]:
        out.append("\nErrors\n")
        out.append(
            table(
                [(r["tool"], r["error_type"], fmt_int(r["n"]), (r["sample"] or "")[:60]) for r in data["errors"]],
                ("TOOL", "TYPE", "N", "SAMPLE"),
                "llrl",
            )
        )
    return "".join(out)


def html_report(data: dict, window: str) -> str:
    e = html.escape
    pct = percentiles(data["latency"])
    t = (data["totals"] or [{}])[0]
    total_tokens = sum(r["tokens"] or 0 for r in data["by_model"])
    known = [r["cost"] for r in data["by_model"] if r["cost"] is not None]

    def rows(items, cells):
        return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in cells(r)) + "</tr>" for r in items)

    day_max = max([r["tokens"] or 0 for r in data["by_day"]] or [1]) or 1
    bars = "".join(
        f'<div class="bar"><div class="fill" style="height:{max(2, round(100 * (r["tokens"] or 0) / day_max))}%"'
        f' title="{e(r["day"])}: {fmt_int(r["tokens"])} tokens"></div><span>{e(r["day"][5:])}</span></div>'
        for r in data["by_day"]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PAL Utilisation</title>
<style>
  :root {{ --bg:#fff; --fg:#16181d; --muted:#6b7280; --line:#e5e7eb; --card:#f9fafb; --accent:#4f46e5; }}
  @media (prefers-color-scheme:dark) {{
    :root {{ --bg:#0f1115; --fg:#e6e8ee; --muted:#9aa1ad; --line:#262b33; --card:#161a20; --accent:#818cf8; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:2rem 1.25rem; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .sub {{ color:var(--muted); margin:0 0 1.75rem; font-size:.9rem; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.75rem; margin-bottom:2rem; }}
  .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }}
  .stat .k {{ color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; }}
  .stat .v {{ font-size:1.5rem; font-weight:600; margin-top:.2rem; font-variant-numeric:tabular-nums; }}
  h2 {{ font-size:1rem; margin:1.75rem 0 .6rem; }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:.875rem; }}
  th,td {{ text-align:right; padding:.45rem .7rem; border-bottom:1px solid var(--line); white-space:nowrap; }}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; font-size:.75rem; text-transform:uppercase; letter-spacing:.04em; }}
  td {{ font-variant-numeric:tabular-nums; }}
  .chart {{ display:flex; align-items:flex-end; gap:6px; height:160px; padding:.5rem 0; }}
  .bar {{ flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; height:100%; }}
  .bar .fill {{ width:100%; background:var(--accent); border-radius:3px 3px 0 0; min-height:2px; }}
  .bar span {{ font-size:.65rem; color:var(--muted); margin-top:.35rem; }}
</style></head><body><div class="wrap">
<h1>PAL Utilisation</h1>
<p class="sub">Last {e(window)} · generated {e(time.strftime('%Y-%m-%d %H:%M'))}</p>
<div class="stats">
  <div class="stat"><div class="k">Tool calls</div><div class="v">{fmt_int(t.get('calls'))}</div></div>
  <div class="stat"><div class="k">Errors</div><div class="v">{fmt_int(t.get('errors'))}</div></div>
  <div class="stat"><div class="k">Tokens</div><div class="v">{fmt_int(total_tokens)}</div></div>
  <div class="stat"><div class="k">Est. cost</div><div class="v">{fmt_cost(sum(known)) if known else '—'}</div></div>
  <div class="stat"><div class="k">Avg latency</div><div class="v">{fmt_int(t.get('avg_ms'))}<span style="font-size:.8rem"> ms</span></div></div>
</div>
<h2>Tokens per day</h2><div class="chart">{bars or '<p class="sub">No data</p>'}</div>
<h2>By tool</h2><div class="scroll"><table>
<tr><th>Tool</th><th>Calls</th><th>Errors</th><th>Tokens</th><th>Avg ms</th><th>p50</th><th>p95</th></tr>
{rows(data['by_tool'], lambda r: (e(r['tool']), fmt_int(r['calls']), fmt_int(r['errors']), fmt_int(r['tokens']),
      fmt_int(r['avg_ms']), fmt_int(pct.get(r['tool'],{}).get('p50')), fmt_int(pct.get(r['tool'],{}).get('p95'))))}
</table></div>
<h2>By model</h2><div class="scroll"><table>
<tr><th>Model</th><th>Provider</th><th>Calls</th><th>In</th><th>Out</th><th>Total</th><th>Cost</th></tr>
{rows(data['by_model'], lambda r: (e(r['model']), e(r['provider']), fmt_int(r['calls']), fmt_int(r['inp']),
      fmt_int(r['outp']), fmt_int(r['tokens']), fmt_cost(r['cost'])))}
</table></div>
<h2>By client</h2><div class="scroll"><table>
<tr><th>Client</th><th>Calls</th></tr>
{rows(data['by_client'], lambda r: (e(r['client']), fmt_int(r['calls'])))}
</table></div>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Report PAL MCP server and model utilisation.")
    ap.add_argument("--last", default="7d", help="Window to report on (e.g. 30m, 24h, 7d, 2w). Default: 7d")
    ap.add_argument("--html", metavar="PATH", help="Also write an HTML dashboard to PATH")
    ap.add_argument("--json", action="store_true", help="Emit raw aggregates as JSON")
    args = ap.parse_args()

    db = get_db_path()
    if not db.exists():
        print(f"No metrics database yet at {db}.\nIt is created on the first tool call after this feature landed.")
        return 1

    since = time.time() - parse_duration(args.last)
    conn = connect(readonly=True)
    try:
        data = gather(conn, since)
    finally:
        conn.close()

    if args.json:
        data.pop("latency", None)
        print(json.dumps(data, indent=2, default=str))
    else:
        print(text_report(data, args.last))

    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_report(data, args.last), encoding="utf-8")
        print(f"Dashboard written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
