# Kawaii Session 2 recognition/rating task

This folder adds Session 2 task files without changing the Session 1 RL task files.

## Files added

- `config/session2_recognition_config.m` — editable configuration with sections and comments.
- `matlab/generate_session2_recognition_schedule.m` — run after Session 1 to generate the subject-specific Session 2 schedule.
- `matlab/run_session2_recognition_task.m` — MATLAB/Psychtoolbox Session 2 task runner.

## Expected workflow

1. Run Session 1 Kawaii RL task.
2. Session 1 should export:
   - `outputs/session2_inputs/{subject}_session1_recognition_pool.csv`
   - `outputs/session2_inputs/{subject}_session1_recognition_pool_summary.txt`
3. In MATLAB, run:
   - `generate_session2_recognition_schedule`
4. Copy the generated schedule/trigger map to the EEG computer if needed:
   - `outputs/session2_schedules/{subject}_session2_recognition_schedule.csv`
   - `outputs/session2_trigger_maps/{subject}_session2_trigger_map.csv`
5. On Session 2 day, run:
   - `run_session2_recognition_task`

## Task structure

- 400 trials total.
- 200 old images, 200 new images.
- 8 blocks × 50 trials.
- Each trial:
  1. optional fixation between trials;
  2. image + recognition question;
  3. same image + kawaii rating;
  4. same image + arousal rating.

No fixation/blank is inserted between the three questions for the same image.

## Defaults

- Left key `s` = YES / move rating cursor left.
- Right key `l` = NO / move rating cursor right.
- `space` = confirm rating / continue.
- Recognition timeout = 4 s.
- Rating timeout = 8 s.
- Between-trial fixation = jittered 0.8–1.2 s.
- Language: English/default (1), German (2).
- Run type: experiment/default (1), debug (2).
- Environment: behavioural/default (1), EEG (2).

## Logs

Logs are saved under:

`outputs/session2_logs/{subject}_session2/`

Main analysis file:

`{subject}_session2_trial_log_analysis.csv`

This is intended to be clean and analysis-friendly, with no redundant duplicate columns.

## EEG triggers

The runner uses numeric triggers following Markus's redundancy scheme:

- image onset old/new: 120/121
- recognition response yes/no: 150/151
- kawaii question onset: 100–109, trial digit 0–9
- kawaii rating response: 11–19
- arousal question onset: 200–239, trial group 0–39
- arousal rating response: 21–29

The EEG trigger implementation follows the RI-SIGA-style `io64`/parallel-port approach. The port address is configured in `config/session2_recognition_config.m`.

## Debug schedule generation after incomplete Session 1 tests

`generate_session2_recognition_schedule.m` now supports two schedule types:

- `experiment/full` (`1`, default): requires the full Session 1 recognition pool with at least 200 old and 200 new images and creates the 400-trial Session 2 schedule.
- `debug/test` (`2`): accepts an incomplete Session 1 debug recognition pool. It uses all available old images up to `cfg.schedule_generation.debug.max_old_new_per_status` and samples the same number of new images, creating a smaller balanced test schedule.

For example, if a short Session 1 debug run showed 12 old images, the debug Session 2 generator can create a 24-trial schedule with 12 old + 12 new images. The default output subject ID for debug schedules is `{subject}_debug`, so the runner should be started with that ID, e.g. `pilot00_debug`.

This is only for technical testing. Real Session 2 data collection should use the full `experiment` schedule type.
