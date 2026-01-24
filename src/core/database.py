"""
Database - SQLite-based persistence for features, templates, and runs
Provides storage for feature metadata, generated prompts, and execution history.
"""
from __future__ import annotations
import sqlite3
import json
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

# Database path - can be configured via environment variable
DB_PATH = Path(__file__).parent.parent / "data" / "metafeature.db"


def _ensure_db_dir():
    """Ensure the data directory exists"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _conn() -> sqlite3.Connection:
    """Get a database connection"""
    _ensure_db_dir()
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _now() -> str:
    """Get current UTC timestamp"""
    return datetime.utcnow().isoformat()


def init_db():
    """Initialize database tables"""
    with _conn() as con:
        # Features table
        con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            group_name TEXT,
            name TEXT NOT NULL,
            category TEXT,
            description TEXT,
            metadata_json TEXT,
            created_at TEXT,
            updated_at TEXT
        )""")
        
        # Prompt templates table
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
            score REAL,
            created_at TEXT,
            FOREIGN KEY (feature_id) REFERENCES features(id)
        )""")
        
        # Evaluation runs table
        con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            feature_id TEXT,
            template_id TEXT,
            language TEXT,
            metrics_json TEXT,
            input_data TEXT,
            output_prompt TEXT,
            result_json TEXT,
            created_at TEXT,
            FOREIGN KEY (feature_id) REFERENCES features(id),
            FOREIGN KEY (template_id) REFERENCES prompt_templates(id)
        )""")
        
        # Create indexes
        con.execute("CREATE INDEX IF NOT EXISTS idx_features_group ON features(group_name)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_features_category ON features(category)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_templates_feature ON prompt_templates(feature_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_feature ON runs(feature_id)")


class FeatureStore:
    """Store and retrieve feature metadata"""
    
    def __init__(self):
        init_db()
    
    def upsert_feature(self, feature_id: str, metadata: Dict[str, Any]) -> str:
        """Insert or update a feature"""
        with _conn() as con:
            existing = con.execute("SELECT id FROM features WHERE id=?", (feature_id,)).fetchone()
            
            if existing:
                con.execute("""
                    UPDATE features SET
                        group_name=?, name=?, category=?, description=?,
                        metadata_json=?, updated_at=?
                    WHERE id=?
                """, (
                    metadata.get("group"),
                    metadata.get("name") or metadata.get("feature_name"),
                    metadata.get("category"),
                    metadata.get("description") or metadata.get("feature_description"),
                    json.dumps(metadata),
                    _now(),
                    feature_id
                ))
            else:
                con.execute("""
                    INSERT INTO features(id, group_name, name, category, description, metadata_json, created_at, updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                """, (
                    feature_id,
                    metadata.get("group"),
                    metadata.get("name") or metadata.get("feature_name"),
                    metadata.get("category"),
                    metadata.get("description") or metadata.get("feature_description"),
                    json.dumps(metadata),
                    _now(),
                    _now()
                ))
        return feature_id
    
    def get_feature(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get a feature by ID"""
        with _conn() as con:
            row = con.execute("SELECT * FROM features WHERE id=?", (feature_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "group": row["group_name"],
                "name": row["name"],
                "category": row["category"],
                "description": row["description"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
    
    def list_features(self, group: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List features with optional filters"""
        with _conn() as con:
            query = "SELECT * FROM features WHERE 1=1"
            params = []
            
            if group:
                query += " AND group_name=?"
                params.append(group)
            if category:
                query += " AND category=?"
                params.append(category)
            
            query += " ORDER BY group_name, name"
            
            rows = con.execute(query, params).fetchall()
            return [
                {
                    "id": row["id"],
                    "group": row["group_name"],
                    "name": row["name"],
                    "category": row["category"],
                    "description": row["description"],
                }
                for row in rows
            ]
    
    def delete_feature(self, feature_id: str) -> bool:
        """Delete a feature"""
        with _conn() as con:
            result = con.execute("DELETE FROM features WHERE id=?", (feature_id,))
            return result.rowcount > 0
    
    def get_groups(self) -> List[str]:
        """Get all unique group names"""
        with _conn() as con:
            rows = con.execute("SELECT DISTINCT group_name FROM features WHERE group_name IS NOT NULL ORDER BY group_name").fetchall()
            return [row["group_name"] for row in rows]


class PromptTemplateStore:
    """Store and retrieve prompt templates"""
    
    def __init__(self):
        init_db()
    
    def upsert_template(
        self,
        feature_id: str,
        language: str,
        category: str,
        metrics: List[str],
        prompt: str,
        source: str,
        score: Optional[float] = None
    ) -> str:
        """Insert a new template version"""
        with _conn() as con:
            # Get next version number
            vrow = con.execute("""
                SELECT COALESCE(MAX(version), 0) AS v FROM prompt_templates
                WHERE feature_id=? AND language=?
            """, (feature_id, language)).fetchone()
            version = int(vrow["v"]) + 1
            
            tid = str(uuid.uuid4())
            con.execute("""
                INSERT INTO prompt_templates(id, feature_id, language, category, metrics_json, prompt_text, source, version, score, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            """, (tid, feature_id, language, category, json.dumps(metrics), prompt, source, version, score, _now()))
            return tid
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID"""
        with _conn() as con:
            row = con.execute("SELECT * FROM prompt_templates WHERE id=?", (template_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "feature_id": row["feature_id"],
                "language": row["language"],
                "category": row["category"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else [],
                "prompt": row["prompt_text"],
                "source": row["source"],
                "version": row["version"],
                "score": row["score"],
                "created_at": row["created_at"],
            }
    
    def get_latest_template(self, feature_id: str, language: str) -> Optional[Dict[str, Any]]:
        """Get the latest template for a feature/language combo"""
        with _conn() as con:
            row = con.execute("""
                SELECT * FROM prompt_templates
                WHERE feature_id=? AND language=?
                ORDER BY version DESC LIMIT 1
            """, (feature_id, language)).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "feature_id": row["feature_id"],
                "language": row["language"],
                "category": row["category"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else [],
                "prompt": row["prompt_text"],
                "source": row["source"],
                "version": row["version"],
                "score": row["score"],
                "created_at": row["created_at"],
            }
    
    def list_templates(self, feature_id: str) -> List[Dict[str, Any]]:
        """List all templates for a feature"""
        with _conn() as con:
            rows = con.execute("""
                SELECT * FROM prompt_templates
                WHERE feature_id=?
                ORDER BY language, version DESC
            """, (feature_id,)).fetchall()
            return [
                {
                    "id": row["id"],
                    "language": row["language"],
                    "version": row["version"],
                    "source": row["source"],
                    "score": row["score"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]


class RunStore:
    """Store and retrieve evaluation runs"""
    
    def __init__(self):
        init_db()
    
    def log_run(
        self,
        feature_id: str,
        template_id: str,
        language: str,
        metrics: List[str],
        output_prompt: str,
        input_data: str = "",
        result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an evaluation run"""
        rid = str(uuid.uuid4())
        with _conn() as con:
            con.execute("""
                INSERT INTO runs(id, feature_id, template_id, language, metrics_json, input_data, output_prompt, result_json, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (
                rid, feature_id, template_id, language,
                json.dumps(metrics), input_data, output_prompt,
                json.dumps(result) if result else None, _now()
            ))
        return rid
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a run by ID"""
        with _conn() as con:
            row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "feature_id": row["feature_id"],
                "template_id": row["template_id"],
                "language": row["language"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else [],
                "input_data": row["input_data"],
                "output_prompt": row["output_prompt"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "created_at": row["created_at"],
            }
    
    def list_runs(self, feature_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent runs"""
        with _conn() as con:
            if feature_id:
                rows = con.execute("""
                    SELECT * FROM runs WHERE feature_id=?
                    ORDER BY created_at DESC LIMIT ?
                """, (feature_id, limit)).fetchall()
            else:
                rows = con.execute("""
                    SELECT * FROM runs ORDER BY created_at DESC LIMIT ?
                """, (limit,)).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "feature_id": row["feature_id"],
                    "template_id": row["template_id"],
                    "language": row["language"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
