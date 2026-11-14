# Dataset compatibility

| Source | Unit | Target | Feasible pilot |
|---|---|---|---|
| Birmingham | Facility | Instant occupied/capacity | Approximate 30/60/120 minutes |
| Harvard SFpark | Segment | Instant API occupied/current capacity | 30/60/120 minutes on bounded sample |
| Melbourne historical | Bay | Binary state within a documented event interval | Retrospective reconstruction tests only |
| SFMTA hourly | Block/hour | Occupied time / known time | Future 60/120-minute interval task after acquisition |

Only independently fitted city-specific models are evaluated. The shared numeric ratio does not establish equal sensing semantics. Capacity varies in Harvard and may be zero during restrictions. Invalid or zero-denominator rows are excluded and counted. Unknown intervals remain unknown in the Melbourne adapter. No cross-city concatenation or transfer was performed.
