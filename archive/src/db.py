import sqlite3
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

DB_PATH = "feature_eval.db"

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with _conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            group_name TEXT,
            name TEXT,
            category TEXT,
            metadata_json TEXT,
            created_at TEXT
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id TEXT PRIMARY KEY,
            feature_id TEXT,
            language TEXT,
            category TEXT,
            metrics_json TEXT,
            prompt_text TEXT,
            source TEXT,
            version INTEGER,
            created_at TEXT
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            feature_id TEXT,
            template_id TEXT,
            language TEXT,
            metrics_json TEXT,
            output_prompt TEXT,
            created_at TEXT
        )""")

def _now():
    return datetime.utcnow().isoformat()

class FeatureStore:
    def __init__(self):
        init_db()

    def upsert_feature(self, feature_id: str, metadata: Dict[str, Any]) -> str:
        with _conn() as con:
            con.execute("""
                INSERT INTO features(id, group_name, name, category, metadata_json, created_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    group_name=excluded.group_name,
                    name=excluded.name,
                    category=excluded.category,
                    metadata_json=excluded.metadata_json
            """, (
                feature_id,
                metadata.get("group"),
                metadata.get("name"),
                metadata.get("category"),
                json.dumps({"metadata": metadata}),
                _now()
            ))
        return feature_id

    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        with _conn() as con:
            row = con.execute("SELECT * FROM features WHERE id=?", (feature_id,)).fetchone()
            if not row:
                return None
            meta = json.loads(row["metadata_json"])
            return {"id": row["id"], **meta}

class PromptTemplateStore:
    def __init__(self):
        init_db()

    def upsert_template(self, feature_id: str, language: str, category: str, metrics: List[str], prompt: str, source: str) -> str:
        import uuid
        with _conn() as con:
            # version increment per feature+language
            vrow = con.execute("""
                SELECT COALESCE(MAX(version),0) AS v FROM prompt_templates
                WHERE feature_id=? AND language=?
            """, (feature_id, language)).fetchone()
            version = int(vrow["v"]) + 1

            tid = str(uuid.uuid4())
            con.execute("""
                INSERT INTO prompt_templates(id, feature_id, language, category, metrics_json, prompt_text, source, version, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (tid, feature_id, language, category, json.dumps(metrics), prompt, source, version, _now()))
            return tid

class RunStore:
    def __init__(self):
        init_db()

    def log_run(self, feature_id: str, template_id: str, language: str, metrics: List[str], output_prompt: str) -> str:
        import uuid
        rid = str(uuid.uuid4())
        with _conn() as con:
            con.execute("""
                INSERT INTO runs(id, feature_id, template_id, language, metrics_json, output_prompt, created_at)
                VALUES(?,?,?,?,?,?,?)
            """, (rid, feature_id, template_id, language, json.dumps(metrics), output_prompt, _now()))
        return rid
