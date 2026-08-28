"""
Measures real Supabase latency from this machine, so we can tell whether the
timeouts are the network, the service, or the app.

Run from the Backend folder:   python measure_supabase.py
"""

import os
import statistics
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_KEY"]
HEADERS = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}

PROBES = [
    ("tiny read  (workspace_settings)", "/rest/v1/workspace_settings?select=name&limit=1"),
    ("small read (sessions)",           "/rest/v1/sessions?select=id&limit=1"),
    ("rpc        (question count)",     "/rest/v1/rpc/count_org_user_questions"),
]

N = 15


def run(label: str, path: str, client: httpx.Client) -> None:
    times, failures = [], 0
    for _ in range(N):
        start = time.perf_counter()
        try:
            if path.endswith("count_org_user_questions"):
                r = client.post(URL + path, headers=HEADERS, json={"org_id": "matz-demo-org"}, timeout=60)
            else:
                r = client.get(URL + path, headers=HEADERS, timeout=60)
            elapsed = time.perf_counter() - start
            if r.status_code >= 400:
                failures += 1
                print(f"    HTTP {r.status_code}: {r.text[:120]}")
            else:
                times.append(elapsed)
        except Exception as e:
            failures += 1
            print(f"    ERR after {time.perf_counter() - start:.1f}s: {type(e).__name__}: {e}")

    print(f"\n{label}")
    if times:
        print(f"  n={len(times)}  min={min(times):.2f}s  median={statistics.median(times):.2f}s  max={max(times):.2f}s")
        slow = [t for t in times if t > 5]
        if slow:
            print(f"  !! {len(slow)}/{len(times)} calls took over 5s — that is the problem")
    if failures:
        print(f"  failures: {failures}/{N}")


print(f"Supabase: {URL}")
print(f"{N} calls per probe, fresh connection pool per probe.\n")

for label, path in PROBES:
    # New client per probe so each one pays its own DNS + TLS setup, matching
    # what a fresh threadpool thread does in the app.
    with httpx.Client() as client:
        run(label, path, client)

print("""
How to read this:
  median under ~0.5s, no slow calls  -> Supabase is fine; the app is the issue
  median fine but occasional 5s+     -> network variance / Supabase instability
  median consistently high           -> distance to your Supabase region
  failures with HTTP 4xx             -> key or permissions problem, not speed
""")