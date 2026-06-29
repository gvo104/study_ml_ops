CREATE TABLE IF NOT EXISTS prediction_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    tokens_stemmed TEXT NOT NULL,
    num_of_characters INTEGER NOT NULL,
    num_of_sentences INTEGER NOT NULL,
    model_prediction TEXT NOT NULL,
    model_confidence REAL,
    model_probabilities_json TEXT NOT NULL,
    expert_label TEXT,
    expert_confidence REAL,
    expert_reason TEXT,
    expert_labeled_at TEXT,
    hidden_phase TEXT,
    hidden_target_label TEXT,
    is_selected_for_expert INTEGER NOT NULL DEFAULT 0,
    is_training_candidate INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_prediction_events_created_at
    ON prediction_events(created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_events_labeled
    ON prediction_events(expert_label, created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_events_training_candidate
    ON prediction_events(is_training_candidate, created_at);
