SELECT COALESCE(c.study, 'clic30-v1') AS study,
       COALESCE(c.dataset, 'clic30-full') AS dataset,
       COALESCE(c.target_bpp, 0.15) AS target_bpp,
       c.image_bytes, c.pixels, c.stream_sha256,
       t.participant, t.source, t.codec_a, t.codec_b, v.choice, v.seconds, v.created AS timestamp_ms
FROM votes v JOIN trials t ON t.id = v.trial
LEFT JOIN trial_context c ON c.trial = t.id
ORDER BY study, t.participant, t.position;
