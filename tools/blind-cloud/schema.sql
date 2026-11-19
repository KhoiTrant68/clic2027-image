CREATE TABLE IF NOT EXISTS invitations (
  digest TEXT PRIMARY KEY,
  participant TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS trials (
  id TEXT PRIMARY KEY,
  participant TEXT NOT NULL REFERENCES invitations(participant),
  position INTEGER NOT NULL,
  source TEXT NOT NULL,
  reference TEXT NOT NULL,
  a TEXT NOT NULL,
  b TEXT NOT NULL,
  codec_a TEXT NOT NULL,
  codec_b TEXT NOT NULL,
  UNIQUE(participant, position)
);
CREATE TABLE IF NOT EXISTS votes (
  trial TEXT PRIMARY KEY REFERENCES trials(id),
  choice TEXT NOT NULL CHECK(choice IN ('A','B','tie','skip')),
  seconds REAL NOT NULL,
  created INTEGER NOT NULL
);
