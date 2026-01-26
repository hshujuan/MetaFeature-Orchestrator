"""Quick database query script"""
import sqlite3
from pathlib import Path

db_path = Path(__file__).parent.parent / "src" / "data" / "metafeature.db"
c = sqlite3.connect(str(db_path))
c.row_factory = sqlite3.Row

print("=== TABLES ===")
for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    print(f"  {t[0]}")

print("\n=== FEATURES ===")
for r in c.execute("SELECT id, name, category FROM features"):
    print(f"  {r['name']} ({r['category']}) - {r['id'][:8]}...")

print("\n=== RECENT TEMPLATES ===")
for r in c.execute("SELECT feature_id, language, version, created_at FROM prompt_templates ORDER BY created_at DESC LIMIT 5"):
    print(f"  v{r['version']} [{r['language']}] - {r['created_at']}")

print("\n=== COUNTS ===")
print(f"  Features: {c.execute('SELECT COUNT(*) FROM features').fetchone()[0]}")
print(f"  Templates: {c.execute('SELECT COUNT(*) FROM prompt_templates').fetchone()[0]}")
print(f"  Runs: {c.execute('SELECT COUNT(*) FROM runs').fetchone()[0]}")

c.close()
