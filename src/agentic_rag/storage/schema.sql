PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY,
  arxiv_id TEXT NOT NULL,
  arxiv_version TEXT,
  title TEXT NOT NULL,
  authors TEXT NOT NULL,
  abstract TEXT NOT NULL,
  primary_category TEXT,
  categories TEXT NOT NULL,
  published_at TEXT,
  updated_at TEXT,
  pdf_url TEXT,
  abs_url TEXT,
  doi TEXT,
  comment TEXT,
  source_query TEXT,
  ingested_at TEXT NOT NULL,
  pdf_sha256 TEXT,
  parse_status TEXT NOT NULL,
  parser_name TEXT,
  parser_version TEXT,
  chunker_name TEXT,
  chunker_config_hash TEXT
);

CREATE TABLE IF NOT EXISTS parent_sections (
  parent_id TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL,
  section_path TEXT NOT NULL,
  section_heading TEXT,
  section_type TEXT,
  section_level INTEGER,
  parent_index INTEGER NOT NULL,
  page_start INTEGER,
  page_end INTEGER,
  token_count INTEGER,
  char_count INTEGER,
  content_types TEXT,
  child_chunk_ids TEXT,
  text_hash TEXT,
  parent_text TEXT NOT NULL,
  FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS child_chunks (
  chunk_id TEXT PRIMARY KEY,
  parent_id TEXT NOT NULL,
  paper_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  section_path TEXT NOT NULL,
  section_type TEXT,
  content_type TEXT,
  page_start INTEGER,
  page_end INTEGER,
  token_count INTEGER,
  char_start INTEGER,
  char_end INTEGER,
  docling_ref_ids TEXT,
  embedding_text_hash TEXT,
  embedding_model TEXT,
  embedding_config_hash TEXT,
  indexed_embedding_text_hash TEXT,
  last_indexed_at TEXT,
  created_at TEXT NOT NULL,
  chunk_text TEXT NOT NULL,
  FOREIGN KEY (parent_id) REFERENCES parent_sections (parent_id) ON DELETE CASCADE,
  FOREIGN KEY (paper_id) REFERENCES papers (paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantic_memories (
  memory_id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  confidence REAL NOT NULL,
  source_turn_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  user_query TEXT NOT NULL,
  route_action TEXT NOT NULL,
  route_confidence REAL,
  retrieved_child_ids TEXT,
  selected_parent_ids TEXT,
  tool_calls TEXT,
  final_action TEXT NOT NULL,
  final_answer_summary TEXT,
  user_feedback TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  run_mode TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT
);

CREATE TABLE IF NOT EXISTS trace_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  trace_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  level TEXT NOT NULL,
  event TEXT NOT NULL,
  node TEXT,
  payload_json TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_traces (
  retrieval_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  top_child_score REAL,
  top_parent_score REAL,
  second_parent_score REAL,
  score_margin REAL,
  supporting_child_count INTEGER,
  distinct_parent_count INTEGER,
  distinct_paper_count INTEGER,
  section_type_distribution TEXT,
  confidence_band TEXT,
  evidence_status TEXT,
  thresholds_used TEXT,
  final_evidence_action TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_traces (
  tool_call_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  tool_args_json TEXT NOT NULL,
  tool_started_at TEXT NOT NULL,
  tool_finished_at TEXT,
  tool_status TEXT NOT NULL,
  tool_latency_ms INTEGER,
  tool_result_summary TEXT,
  tool_error TEXT,
  tool_result_ref TEXT,
  FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_traces (
  evidence_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  evidence_status TEXT NOT NULL,
  confidence_band TEXT,
  missing_info TEXT,
  contradiction_notes TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answer_traces (
  answer_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  final_action TEXT NOT NULL,
  citation_ids TEXT,
  refusal_reason TEXT,
  clarifying_question TEXT,
  answer_text TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eval_cases (
  case_id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  conversation_history TEXT,
  expected_route TEXT,
  expected_final_action TEXT,
  expected_tool TEXT,
  expected_paper_ids TEXT,
  expected_parent_ids TEXT,
  expected_answer_points TEXT,
  should_cite_sources INTEGER NOT NULL DEFAULT 1,
  should_refuse INTEGER NOT NULL DEFAULT 0,
  should_clarify INTEGER NOT NULL DEFAULT 0,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS eval_runs (
  eval_run_id TEXT PRIMARY KEY,
  variant TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  total_cases INTEGER NOT NULL,
  raw_score REAL,
  normalized_score REAL,
  hard_fail_refusal INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS eval_scores (
  score_id TEXT PRIMARY KEY,
  eval_run_id TEXT NOT NULL,
  case_id TEXT NOT NULL,
  score REAL NOT NULL,
  route_score REAL,
  retrieval_score REAL,
  evidence_score REAL,
  answer_score REAL,
  citation_score REAL,
  memory_score REAL,
  tool_score REAL,
  trace_score REAL,
  notes TEXT,
  FOREIGN KEY (eval_run_id) REFERENCES eval_runs (eval_run_id) ON DELETE CASCADE,
  FOREIGN KEY (case_id) REFERENCES eval_cases (case_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers (arxiv_id);
CREATE INDEX IF NOT EXISTS idx_parent_sections_paper_id ON parent_sections (paper_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id ON child_chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_paper_id ON child_chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_trace_events_trace_id ON trace_events (trace_id);
CREATE INDEX IF NOT EXISTS idx_tool_traces_trace_id ON tool_traces (trace_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_traces_trace_id ON retrieval_traces (trace_id);
