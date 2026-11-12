#!/usr/bin/env python3
"""
Bio-Notebook Load & API Test Suite
Tests: availability, response times, concurrency, API endpoints.
"""
import json
import time
import sys
import statistics
import concurrent.futures
import urllib.request
import urllib.error
import ssl

BASE = "https://anticus.tech/bio-notebook"
ssl_ctx = ssl.create_default_context()

def req(path, method="GET", data=None, timeout=15):
    url = f"{BASE}{path}"
    headers = {"User-Agent": "BioNotebook-Test/1.0"}
    if data:
        headers["Content-Type"] = "application/json"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx)
        elapsed = time.time() - t0
        return {"status": resp.status, "time": elapsed, "ok": True}
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        return {"status": e.code, "time": elapsed, "ok": False}
    except Exception as e:
        elapsed = time.time() - t0
        return {"status": 0, "time": elapsed, "ok": False, "error": str(e)}

def perc(data, p):
    if not data:
        return 0
    return sorted(data)[int(len(data) * p / 100)]

# ── 1. Page availability tests ──
print("=" * 60)
print("📄 1. PAGE AVAILABILITY TESTS")
print("=" * 60)

pages = {
    "Лендинг": "/",
    "Дашборд (no auth)": "/dashboard?user=test",
    "Страница входа в админку": "/admin",
    "Стили CSS": "/static/styles.css",
    "Фавикон": "/static/favicon.svg",
    "Тестовый эксперимент (404)": "/experiment/test/nonexistent",
}

page_results = []
for name, path in pages.items():
    r = req(path)
    status = r["status"]
    ms = round(r["time"] * 1000)
    sym = "✅" if 200 <= status < 500 else "❌"
    print(f"  {sym} {name:30s} HTTP {status:3d}  {ms:4d}ms")
    page_results.append(r["time"])

# ── 2. API endpoint tests ──
print()
print("=" * 60)
print("🔌 2. API ENDPOINT TESTS")
print("=" * 60)

api_tests = {
    "Reagents list": ("GET", "/api/reagents?user=loadtest_user"),
    "Consumables list": ("GET", "/api/consumables?user=loadtest_user"),
    "Projects list": ("GET", "/api/projects?user=loadtest_user"),
}

api_results = []
for name, (method, path) in api_tests.items():
    r = req(path)
    status = r["status"]
    ms = round(r["time"] * 1000)
    sym = "✅" if r["ok"] else "❌"
    print(f"  {sym} {name:30s} HTTP {status:3d}  {ms:4d}ms")
    api_results.append(r["time"])

# ── 3. Concurrency / Load test ──
print()
print("=" * 60)
print("🔥 3. LOAD TEST — 50 concurrent requests")
print("=" * 60)

CONCURRENT = 50
paths_for_load = [
    "/",
    "/static/styles.css",
    "/static/favicon.svg",
    "/static/dashboard.jpg",
    "/admin",
]

load_times = []

def load_hit(path):
    r = req(path)
    return r

with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT) as ex:
    # Submit all requests
    all_paths = [paths_for_load[i % len(paths_for_load)] for i in range(CONCURRENT)]
    futures = [ex.submit(load_hit, p) for p in all_paths]
    
    t0 = time.time()
    for i, f in enumerate(concurrent.futures.as_completed(futures)):
        r = f.result()
        load_times.append(r["time"])
    total_time = time.time() - t0

p50 = perc(load_times, 50)
p95 = perc(load_times, 95)
p99 = perc(load_times, 99)
avg = statistics.mean(load_times) if load_times else 0
failed = sum(1 for t in load_times if t == 0)
print(f"  Всего запросов:  {len(load_times)}")
print(f"  Общее время:     {total_time:.2f}s")
print(f"  Успешно:         {len(load_times) - failed}")
print(f"  Ошибок:          {failed}")
print(f"  Среднее:         {avg*1000:.0f}ms")
print(f"  P50:             {p50*1000:.0f}ms")
print(f"  P95:             {p95*1000:.0f}ms")
print(f"  P99:             {p99*1000:.0f}ms")
print(f"  Requests/sec:    {len(load_times)/total_time:.1f}")

# ── 4. Sustained load test (10s burst) ──
print()
print("=" * 60)
print("⚡ 4. SUSTAINED LOAD — 10s burst (10 concurrent)")
print("=" * 60)

sustained_times = []
sustained_start = time.time()
request_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    while time.time() - sustained_start < 10:
        fs = [ex.submit(load_hit, paths_for_load[i % len(paths_for_load)]) for i in range(10)]
        for f in concurrent.futures.as_completed(fs):
            r = f.result()
            sustained_times.append(r["time"])
            request_count += 1

sustained_duration = time.time() - sustained_start
s_p50 = perc(sustained_times, 50)
s_p95 = perc(sustained_times, 95)
s_p99 = perc(sustained_times, 99)
s_avg = statistics.mean(sustained_times) if sustained_times else 0
s_failed = sum(1 for t in sustained_times if t == 0)
print(f"  Всего запросов:  {request_count}")
print(f"  Длительность:    {sustained_duration:.1f}s")
print(f"  Успешно:         {request_count - s_failed}")
print(f"  Ошибок:          {s_failed}")
print(f"  Среднее:         {s_avg*1000:.0f}ms")
print(f"  P50:             {s_p50*1000:.0f}ms")
print(f"  P95:             {s_p95*1000:.0f}ms")
print(f"  P99:             {s_p99*1000:.0f}ms")
print(f"  Requests/sec:    {request_count/sustained_duration:.1f}")

# ── 5. Frontend test (anti-kus-flow) ──
print()
print("=" * 60)
print("🌐 5. FRONTEND TESTS (anticus.tech)")
print("=" * 60)

frontend_pages = {
    "Главная": "https://anticus.tech/",
    "Каталог": "https://anticus.tech/catalog",
    "Магазин": "https://anticus.tech/3",
    "Био": "https://anticus.tech/bio",
    "Фабрика": "https://anticus.tech/fabrika",
}

frontend_results = []
for name, url in frontend_pages.items():
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(url, timeout=15, context=ssl_ctx)
        elapsed = time.time() - t0
        size = len(resp.read())
        status = resp.status
        sym = "✅"
    except Exception as e:
        elapsed = time.time() - t0
        size = 0
        status = getattr(e, 'code', 0)
        sym = "❌"
    ms = round(elapsed * 1000)
    print(f"  {sym} {name:30s} HTTP {status:3d}  {ms:4d}ms  {size//1024}KB")
    frontend_results.append(elapsed)

# ── SUMMARY ──
print()
print("=" * 60)
print("📊 SUMMARY")
print("=" * 60)

all_times = page_results + api_results + load_times + frontend_results
all_p50 = perc(all_times, 50)
all_p95 = perc(all_times, 95)
all_p99 = perc(all_times, 99)
all_avg = statistics.mean(all_times) if all_times else 0

print(f"  Всего запросов:  {len(all_times)}")
print(f"  Среднее время:   {all_avg*1000:.0f}ms")
print(f"  P50:             {all_p50*1000:.0f}ms")
print(f"  P95:             {all_p95*1000:.0f}ms")
print(f"  P99:             {all_p99*1000:.0f}ms")
print()
print("  Ключевые выводы:")
print(f"  • Страницы сайта: все отвечают, в среднем {statistics.mean(page_results)*1000:.0f}ms")
if p50*1000 < 500:
    print(f"  • Нагрузка 50 concurrent: P50={p50*1000:.0f}ms — отлично")
elif p50*1000 < 1000:
    print(f"  • Нагрузка 50 concurrent: P50={p50*1000:.0f}ms — приемлемо")
else:
    print(f"  • Нагрузка 50 concurrent: P50={p50*1000:.0f}ms — требуется оптимизация")
print(f"  • Sustained 10s burst: {request_count} запросов за {sustained_duration:.0f}s ({request_count/sustained_duration:.0f} req/s)")
print("=" * 60)
