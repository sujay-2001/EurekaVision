#!/usr/bin/env python3
"""Scrape Gymnasium environment documentation pages.

Given an *engine* (e.g. "classic_control", "mujoco", "box2d") and an *env*
(e.g. "cart_pole", "reacher"), this script downloads the documentation page
from
    https://gymnasium.farama.org/environments/{engine}/{env}/
extracts the **Description**, **Action Space** (first paragraph) and the
**Observation Space** table, and writes two files in the chosen output
folder:

* ``{env}_obs.json`` – machine‑readable version of the information.
* ``{env}_obs.txt``  – Markdown‑friendly text with tables preserved.

Dependencies: ``requests`` and ``beautifulsoup4``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Dict

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


def _fetch_html(url: str) -> str:
    """Download *url* and return HTML text (raises for HTTP errors)."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def _section_text(soup: BeautifulSoup, section_id: str) -> str:
    """Return plain text of the top‑level <p> elements in ``<section id=...>``."""
    sec = soup.find("section", id=section_id)
    if not sec:
        return ""
    parts: List[str] = [p.get_text(" ", strip=True) for p in sec.find_all("p", recursive=False)]
    return "\n\n".join(parts).strip()


def _parse_observation_table(soup: BeautifulSoup) -> List[Dict[str, str]]:
    sec = soup.find("section", id="observation-space")
    if not sec:
        return []
    tbl = sec.find("table")
    if not tbl:
        return []

    # Header cells
    head_cells = [th.get_text(" ", strip=True) for th in tbl.thead.find_all("th")]

    rows: List[Dict[str, str]] = []
    for tr in tbl.tbody.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(dict(zip(head_cells, cells)))
    return rows


def _markdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    md_lines = ["| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    return "\n".join(md_lines)


def scrape(engine: str, env: str, out_dir: pathlib.Path) -> None:
    url = f"https://gymnasium.farama.org/environments/{engine.strip('/')}/{env.strip('/')}/"
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    description = _section_text(soup, "description")
    action_space = _section_text(soup, "action-space")
    observation_rows = _parse_observation_table(soup)
    observation_space = []
    for row in observation_rows:
        obs = {k.lower(): v for k, v in row.items()}
        observation_space.append(obs)
    data = {
        "Description": description,
        "Action Space": action_space,
        "Observation Space": observation_space,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_file = out_dir / f"{env}_obs.json"
    txt_file = out_dir / f"{env}_obs.txt"

    with json_file.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

    with txt_file.open("w", encoding="utf-8") as fp:
        fp.write("### Description\n" + description + "\n\n")
        fp.write("### Action Space\n" + action_space + "\n\n")
        fp.write("### Observation Space\n")
        fp.write(_markdown_table(observation_rows))

    print(f"Saved → {json_file}\nSaved → {txt_file}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape Gymnasium environment documentation.")
    parser.add_argument("engine", help="Engine family: classic_control, mujoco, box2d, …")
    parser.add_argument("env", help="Environment slug: cart_pole, reacher, …")
    parser.add_argument("-o", "--out", default=pathlib.Path.cwd(), type=pathlib.Path,
                        help="Output directory (default: current working dir)")

    args = parser.parse_args(argv)
    try:
        scrape(args.engine, args.env, args.out.expanduser().resolve())
    except requests.HTTPError as exc:
        sys.exit(f"HTTP error: {exc.response.status_code} {exc.response.reason}")
    except Exception as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
