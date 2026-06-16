-- Schema for RAG curriculum metadata, ACLs, and audit log
-- Initialised on first Postgres container start

CREATE TABLE IF NOT EXISTS subjects (
    slug VARCHAR(128) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    content_type VARCHAR(32) NOT NULL,
    parser VARCHAR(32) NOT NULL,
    acl_groups TEXT[] NOT NULL DEFAULT ARRAY['all_students'],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_ingested_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    subject_slug VARCHAR(128) NOT NULL REFERENCES subjects(slug),
    unit_number INT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    page_count INT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (subject_slug, unit_number, file_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_subject ON documents(subject_slug);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(128) PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_chunk_id VARCHAR(128),
    subject_slug VARCHAR(128) NOT NULL,
    unit_number INT NOT NULL,
    page_number INT,
    section_path TEXT,
    chunk_type VARCHAR(32) NOT NULL DEFAULT 'child',
    token_count INT,
    content TEXT NOT NULL,
    context_prefix TEXT,
    acl_groups TEXT[] NOT NULL DEFAULT ARRAY['all_students'],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_subject ON chunks(subject_slug);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(128) UNIQUE NOT NULL,
    display_name VARCHAR(256),
    acl_groups TEXT[] NOT NULL DEFAULT ARRAY['all_students'],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_log (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    query TEXT NOT NULL,
    subject_filter VARCHAR(128),
    retrieved_chunk_ids TEXT[],
    rerank_scores REAL[],
    answer TEXT,
    latency_ms INT,
    user_feedback VARCHAR(16),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_log_user ON query_log(user_id);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);

-- Seed two demo users to showcase multi-tenancy / ACL filtering
INSERT INTO users (username, display_name, acl_groups) VALUES
    ('demo_student', 'Demo Student (all subjects)', ARRAY['all_students']),
    ('ml_student', 'ML Track Student', ARRAY['ml_track'])
ON CONFLICT (username) DO NOTHING;
