SELECT t.participant, t.source, t.codec_a, t.codec_b, v.choice, v.seconds, v.created AS timestamp_ms
FROM votes v JOIN trials t ON t.id = v.trial
ORDER BY t.participant, t.position;
