# Data dictionary

| Birmingham field | Meaning and use |
|---|---|
| SystemCodeNumber | Car park key. A site identifier can group observations but does not identify individual hardware. |
| Capacity | Site capacity in spaces, positive and constant within each observed park. |
| Occupancy | Recorded count in spaces. Values outside [0,Capacity] fail a physical constraint. |
| LastUpdated | Recorded observation timestamp, second precision, no supplied offset or receipt timestamp. |
| rate, target | Occupancy / Capacity on valid unique records; target is a subsequent measured ratio at the same park. |
| time, target_time, nominal_target_time | Forecast issue time, actual matched reading time, and issue time plus specified horizon. Only nominal target calendar is available in advance. |

Melbourne uses a semicolon-delimited file. lastupdated and status_timestamp contain offset-bearing update and status times. zone_number denotes a zone, kerbsideid a bay, location a latitude/longitude pair, status_description a recorded bay state. Local counts are 6,324 unique bay IDs in 6,324 rows, 3,383 Unoccupied and 2,941 Present. No site capacity or aggregate occupancy count is supplied.

In the experiment current_input is the latest valid observed ratio carried forward within a park. previous_input and older_input lag that stream. age is minutes since the source of the most recent usable value. missing and invalid flag the arriving input. completeness is the valid fraction of the last three observed rows including the origin. past_validity summarizes up to twelve previous rows. jump is the absolute change in the carried ratio. quality is the bounded heuristic defined in methodology.md. injected is an experiment-only indicator used for evaluation. All error and interval outputs use fractions of capacity. severity is the nominal injection parameter; injected_share records realized exposure.
