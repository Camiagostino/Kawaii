function generate_session2_recognition_schedule()
%GENERATE_SESSION2_RECOGNITION_SCHEDULE Create Session 2 recognition/rating schedule.
%
% Run after Session 1. It reads:
%   outputs/session2_inputs/{subject}_session1_recognition_pool.csv
% and writes:
%   outputs/session2_schedules/{subject}_session2_recognition_schedule.csv
%   outputs/session2_trigger_maps/{subject}_session2_trigger_map.csv
%
% It does not modify Session 1 scripts or schedules.
%
% The generator supports two schedule types:
%   experiment/full = capped by config; can fill missing old images with extra new images
%   debug/test      = accepts incomplete Session 1 debug pools and writes a
%                     small balanced old/new schedule for technical testing

    clc;
    projectRoot = findProjectRoot();
    addpath(fullfile(projectRoot, 'config'));
    cfg = session2_recognition_config();

    defaults = {'1', '1'};
    prompt = { ...
        'Subject ID of Session 1 input (e.g. 1, sub001, pilot00):', ...
        'Schedule type: experiment/full (default/1) or debug/test (2):'};
    answer = inputdlg(prompt, 'Generate Session 2 recognition schedule', [1 80], defaults);
    if isempty(answer); disp('Cancelled.'); return; end

    inputSubject = normaliseSubjectId(strtrim(answer{1}));
    if strlength(inputSubject) == 0; error('Subject ID cannot be empty.'); end
    session1 = '1';
    scheduleType = parseScheduleType(answer{2}, cfg);
    outputSubject = defaultOutputSubject(inputSubject, scheduleType, cfg);
    seed = cfg.task.random_seed_base + deterministicSubjectNumber(char(outputSubject));

    inputPath = findSession1RecognitionPool(projectRoot, cfg, char(inputSubject), session1);

    pool = readtable(inputPath, 'TextType', 'string');
    pool = validateRecognitionPool(pool, cfg, scheduleType);

    rng(seed, 'twister');

    schedule = buildSession2Schedule(pool, cfg, char(outputSubject), seed, scheduleType);
    triggerMap = buildTriggerMap(schedule, cfg);

    outSubject = char(outputSubject);
    scheduleDir = fullfile(projectRoot, cfg.paths.session2_schedule_dir);
    triggerDir = fullfile(projectRoot, cfg.paths.session2_trigger_map_dir);
    if ~exist(scheduleDir, 'dir'); mkdir(scheduleDir); end
    if ~exist(triggerDir, 'dir'); mkdir(triggerDir); end

    [outSubject, schedulePath, triggerPath] = resolveScheduleOutputPaths(projectRoot, cfg, outSubject);
    if ~strcmp(outSubject, char(outputSubject))
        schedule.subject_id(:) = string(outSubject);
        triggerMap.subject_id(:) = string(outSubject);
    end

    writetable(schedule, schedulePath);
    writetable(triggerMap, triggerPath);
    summaryPath = strrep(schedulePath, '.csv', '_summary.txt');
    writeScheduleSummary(summaryPath, inputPath, schedulePath, triggerPath, schedule, cfg, seed, scheduleType, char(inputSubject));

    fprintf('\nSession 2 schedule generated.\n');
    fprintf('Input subject: %s\n', char(inputSubject));
    fprintf('Schedule type: %s\n', scheduleType);
    fprintf('Subject/log ID: %s\n', outSubject);
    fprintf('Schedule: %s\n', schedulePath);
    fprintf('Trigger map: %s\n', triggerPath);
    fprintf('Summary: %s\n', summaryPath);
end

function projectRoot = findProjectRoot()
    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(here);
    if ~exist(fullfile(projectRoot, 'config'), 'dir')
        error('Could not find project root from MATLAB script location.');
    end
end

function subject = normaliseSubjectId(raw)
    raw = string(strtrim(char(raw)));
    if strlength(raw) == 0
        subject = raw; return;
    end
    if ~isempty(regexp(char(raw), '^\d+$', 'once'))
        subject = string(sprintf('sub%03d', str2double(raw)));
    else
        subject = raw;
    end
end

function inputPath = findSession1RecognitionPool(projectRoot, cfg, subject, session1)
    cleanName = [subject '_session' session1 '_recognition_pool.csv'];
    matlabName = [subject '_session' session1 '_matlab_recognition_pool.csv'];
    candidates = {fullfile(projectRoot, cfg.paths.session2_input_dir, cleanName), ...
                  fullfile(projectRoot, cfg.paths.session2_input_dir, matlabName)};
    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file')
            inputPath = candidates{i};
            return;
        end
    end
    error('Session 2 input pool not found. Tried:\n%s\n%s\n\nRun Session 1 first and make sure it exported the recognition pool.', candidates{1}, candidates{2});
end

function scheduleType = parseScheduleType(raw, cfg)
    raw = lower(strtrim(char(string(raw))));
    if isempty(raw) || strcmp(raw, '1') || strcmp(raw, 'experiment') || strcmp(raw, 'full')
        scheduleType = 'experiment';
    elseif strcmp(raw, '2') || strcmp(raw, 'debug') || strcmp(raw, 'test')
        scheduleType = 'debug';
    else
        error('Unknown schedule type: %s. Use 1/experiment or 2/debug.', raw);
    end
    if ~isfield(cfg.schedule_generation, scheduleType)
        error('Schedule type %s is not configured in cfg.schedule_generation.', scheduleType);
    end
end

function outputSubject = defaultOutputSubject(inputSubject, scheduleType, cfg)
    inputSubject = string(inputSubject);
    suffix = '';
    if isfield(cfg.schedule_generation, scheduleType) && isfield(cfg.schedule_generation.(scheduleType), 'output_subject_suffix')
        suffix = char(cfg.schedule_generation.(scheduleType).output_subject_suffix);
    end
    outputSubject = string([char(inputSubject) suffix]);
end

function n = deterministicSubjectNumber(subject)
    nums = regexp(subject, '\d+', 'match');
    if ~isempty(nums)
        n = str2double(nums{end});
    else
        bytes = double(char(subject));
        n = sum(bytes .* (1:numel(bytes)));
    end
    if isnan(n); n = 0; end
end

function pool = validateRecognitionPool(pool, cfg, scheduleType)
    needed = {'image_filename','image_path','old_new_status'};
    for i = 1:numel(needed)
        if ~ismember(needed{i}, pool.Properties.VariableNames)
            error('Recognition pool is missing required column: %s', needed{i});
        end
    end
    pool.old_new_status = lower(string(pool.old_new_status));
    oldN = sum(pool.old_new_status == "old");
    newN = sum(pool.old_new_status == "new");
    if oldN < 1 || newN < 1
        error('%s schedule requires at least 1 old and 1 new image. Found old=%d, new=%d.', scheduleType, oldN, newN);
    end

    maxTotal = getScheduleMaxTotalTrials(cfg, scheduleType);
    validateEvenPositive(maxTotal, sprintf('cfg.schedule_generation.%s.max_total_trials', scheduleType));
    if strcmp(scheduleType, 'experiment') && maxTotal > 400
        error('cfg.schedule_generation.experiment.max_total_trials must not exceed 400 unless trigger coding is redesigned.');
    end
    [plannedOld, plannedNew] = plannedOldNewCounts(oldN, newN, maxTotal, scheduleType, cfg);
    actualTotal = plannedOld + plannedNew;

    if strcmp(scheduleType, 'experiment')
        if actualTotal < maxTotal
            warning(['Session 2 schedule will use %d trials (%d old + %d new) because only old=%d/new=%d images are available, ', ...
                     'or because max_total_trials=%d.'], actualTotal, plannedOld, plannedNew, oldN, newN, maxTotal);
        elseif plannedOld ~= plannedNew
            fprintf('Session 2 schedule will keep %d trials using %d old + %d new images.\n', actualTotal, plannedOld, plannedNew);
        end
        if isfield(cfg.schedule_generation.experiment, 'min_total_warning') && actualTotal < cfg.schedule_generation.experiment.min_total_warning
            warning('Session 2 schedule has only %d trials, below min_total_warning=%d.', actualTotal, cfg.schedule_generation.experiment.min_total_warning);
        end
    end
end

function schedule = buildSession2Schedule(pool, cfg, subject, seed, scheduleType)
    oldRows = pool(pool.old_new_status == "old", :);
    newRows = pool(pool.old_new_status == "new", :);
    availableOld = height(oldRows);
    availableNew = height(newRows);

    maxTotal = getScheduleMaxTotalTrials(cfg, scheduleType);
    validateEvenPositive(maxTotal, sprintf('cfg.schedule_generation.%s.max_total_trials', scheduleType));

    if strcmp(scheduleType, 'experiment')
        plannedBlocks = cfg.schedule_generation.experiment.n_blocks;
        validatePositiveInteger(plannedBlocks, 'cfg.schedule_generation.experiment.n_blocks');
        plannedTrialsPerBlock = ceil(maxTotal / plannedBlocks);
    else
        plannedTrialsPerBlock = cfg.schedule_generation.debug.trials_per_block;
        validatePositiveInteger(plannedTrialsPerBlock, 'cfg.schedule_generation.debug.trials_per_block');
        plannedBlocks = ceil(maxTotal / plannedTrialsPerBlock);
    end

    [nOld, nNew] = plannedOldNewCounts(availableOld, availableNew, maxTotal, scheduleType, cfg);
    actualTotal = nOld + nNew;

    oldRows = oldRows(randperm(height(oldRows), nOld), :);
    newRows = newRows(randperm(height(newRows), nNew), :);
    rows = [oldRows; newRows];
    rows = rows(randperm(height(rows)), :);
    rows = enforceOldNewStreak(rows, cfg.task.max_consecutive_old_new);

    n = height(rows);
    if n ~= actualTotal
        error('Internal error: expected %d schedule rows, got %d.', actualTotal, n);
    end

    trial = (1:n)';
    trial0 = trial - 1;
    block = ceil(trial ./ plannedTrialsPerBlock);
    trialInBlock = trial - (block - 1) * plannedTrialsPerBlock;
    trialDigit = mod(trial0, 10);
    trialGroup = floor(trial0 ./ 10);

    fixation = makeFixationDurations(cfg, n, seed);

    schedule = table();
    schedule.subject_id = repmat(string(subject), n, 1);
    schedule.session = repmat(cfg.task.session2_id, n, 1);
    schedule.schedule_type = repmat(string(scheduleType), n, 1);
    schedule.available_old_images = repmat(availableOld, n, 1);
    schedule.available_new_images = repmat(availableNew, n, 1);
    schedule.n_old_used = repmat(nOld, n, 1);
    schedule.n_new_used = repmat(nNew, n, 1);
    schedule.old_new_balance = repmat(string(sprintf('%d old + %d new', nOld, nNew)), n, 1);
    schedule.max_total_trials_configured = repmat(maxTotal, n, 1);
    schedule.planned_n_blocks_configured = repmat(plannedBlocks, n, 1);
    schedule.planned_trials_per_block = repmat(plannedTrialsPerBlock, n, 1);
    schedule.actual_total_trials = repmat(n, n, 1);
    schedule.actual_n_blocks = repmat(numel(unique(block)), n, 1);
    schedule.block = block;
    schedule.trial = trial;
    schedule.trial_in_block = trialInBlock;
    schedule.trial_index_0_based = trial0;
    schedule.trial_digit_0_9 = trialDigit;
    schedule.trial_group_0_39 = trialGroup;
    schedule.image_filename = string(rows.image_filename);
    schedule.image_path = string(rows.image_path);
    schedule.old_new_status = string(rows.old_new_status);
    schedule.correct_recognition_response = repmat("no", n, 1);
    schedule.correct_recognition_response(schedule.old_new_status == "old") = "yes";
    schedule.recognition_left_label = repmat("YES", n, 1);
    schedule.recognition_right_label = repmat("NO", n, 1);
    schedule.recognition_left_response = repmat("yes", n, 1);
    schedule.recognition_right_response = repmat("no", n, 1);
    schedule.fixation_duration_s = fixation;
    schedule.recognition_timeout_s = repmat(cfg.timeouts.recognition_s, n, 1);
    schedule.kawaii_timeout_s = repmat(cfg.timeouts.rating_s, n, 1);
    schedule.arousal_timeout_s = repmat(cfg.timeouts.rating_s, n, 1);
    schedule.trigger_image_onset = repmat(cfg.triggers.image_new, n, 1);
    schedule.trigger_image_onset(schedule.old_new_status == "old") = cfg.triggers.image_old;
    schedule.trigger_recognition_yes = repmat(cfg.triggers.recognition_yes, n, 1);
    schedule.trigger_recognition_no = repmat(cfg.triggers.recognition_no, n, 1);
    schedule.trigger_kawaii_question = cfg.triggers.kawaii_question_base + trialDigit;
    schedule.trigger_arousal_question = cfg.triggers.arousal_question_base + trialGroup;
    schedule.trigger_kawaii_response_base = repmat(cfg.triggers.kawaii_response_base, n, 1);
    schedule.trigger_arousal_response_base = repmat(cfg.triggers.arousal_response_base, n, 1);
    schedule.trigger_label_image_onset = strcat(upper(extractBefore(schedule.old_new_status, 2)), string(schedule.trigger_image_onset));

    optional = {'source_condition','source_run','source_trial','source_block','source_rule_segment','source_feedback_type', ...
        'M_Kawaii','M_Arousal','species','mammal_status','animal_class','shown_in_session1','was_shown_in_session1'};
    for i = 1:numel(optional)
        col = optional{i};
        if ismember(col, rows.Properties.VariableNames)
            schedule.(col) = rows.(col);
        end
    end
end

function [nOld, nNew] = plannedOldNewCounts(availableOld, availableNew, maxTotal, scheduleType, cfg)
    maxPerStatus = maxTotal / 2;
    if strcmp(scheduleType, 'experiment')
        strategy = 'balanced_adaptive';
        if isfield(cfg.schedule_generation.experiment, 'total_trial_strategy')
            strategy = lower(strtrim(char(string(cfg.schedule_generation.experiment.total_trial_strategy))));
        end
        switch strategy
            case 'fill_with_new'
                nOld = min(availableOld, maxPerStatus);
                nNew = min(availableNew, maxTotal - nOld);
                % If there are not enough new images to fill the total, use as many
                % additional old images as possible without exceeding maxTotal.
                if nOld + nNew < maxTotal && availableOld > nOld
                    extraOld = min(availableOld - nOld, maxTotal - nOld - nNew);
                    nOld = nOld + extraOld;
                end
            case 'balanced_adaptive'
                nPerStatus = min([availableOld, availableNew, maxPerStatus]);
                nOld = nPerStatus;
                nNew = nPerStatus;
            otherwise
                error('Unknown total_trial_strategy: %s. Use fill_with_new or balanced_adaptive.', strategy);
        end
    else
        nPerStatus = min([availableOld, availableNew, maxPerStatus]);
        nOld = nPerStatus;
        nNew = nPerStatus;
    end
    nOld = floor(nOld);
    nNew = floor(nNew);
    if nOld < 1 || nNew < 1
        error('Cannot build Session 2 schedule: planned old=%d, new=%d.', nOld, nNew);
    end
end

function maxTotal = getScheduleMaxTotalTrials(cfg, scheduleType)
    if isfield(cfg.schedule_generation, scheduleType) && isfield(cfg.schedule_generation.(scheduleType), 'max_total_trials')
        maxTotal = cfg.schedule_generation.(scheduleType).max_total_trials;
    else
        error('Missing cfg.schedule_generation.%s.max_total_trials.', scheduleType);
    end
end

function validateEvenPositive(value, label)
    validatePositiveInteger(value, label);
    if mod(value, 2) ~= 0
        error('%s must be an even integer for predictable block sizes. Current value: %g', label, value);
    end
end

function validatePositiveInteger(value, label)
    if isempty(value) || ~isnumeric(value) || ~isscalar(value) || isnan(value) || value <= 0 || round(value) ~= value
        error('%s must be a positive integer. Current value: %g', label, value);
    end
end

function rows = enforceOldNewStreak(rows, maxRun)
    if isempty(maxRun) || maxRun <= 0; return; end
    best = rows;
    for attempt = 1:1000
        candidate = rows(randperm(height(rows)), :);
        if maxConsecutive(candidate.old_new_status) <= maxRun
            rows = candidate; return;
        end
        if maxConsecutive(candidate.old_new_status) < maxConsecutive(best.old_new_status)
            best = candidate;
        end
    end
    rows = best;
    warning('Could not fully enforce max old/new streak of %d. Best streak: %d.', maxRun, maxConsecutive(best.old_new_status));
end

function m = maxConsecutive(values)
    values = string(values);
    m = 0; current = 0; last = "";
    for i = 1:numel(values)
        if values(i) == last
            current = current + 1;
        else
            current = 1; last = values(i);
        end
        m = max(m, current);
    end
end

function fixation = makeFixationDurations(cfg, n, seed)
    mode = lower(string(cfg.fixation.mode));
    if ~cfg.fixation.enabled || mode == "none"
        fixation = zeros(n, 1);
    elseif mode == "fixed"
        fixation = repmat(cfg.fixation.fixed_s, n, 1);
    elseif mode == "jittered"
        rng(seed + 999, 'twister');
        fixation = cfg.fixation.min_s + (cfg.fixation.max_s - cfg.fixation.min_s) .* rand(n, 1);
        fixation = round(fixation, 3);
    else
        error('Unknown fixation mode: %s', mode);
    end
end

function [outSubject, schedulePath, triggerPath] = resolveScheduleOutputPaths(projectRoot, cfg, subject)
    outSubject = subject;
    scheduleDir = fullfile(projectRoot, cfg.paths.session2_schedule_dir);
    triggerDir = fullfile(projectRoot, cfg.paths.session2_trigger_map_dir);
    schedulePath = fullfile(scheduleDir, [outSubject cfg.paths.session2_schedule_suffix]);
    triggerPath = fullfile(triggerDir, [outSubject cfg.paths.session2_trigger_map_suffix]);
    if ~exist(schedulePath, 'file') && ~exist(triggerPath, 'file'); return; end

    choice = questdlg(sprintf('Session 2 schedule/trigger map already exists for %s. Save as next suffix instead?', subject), ...
        'Existing Session 2 schedule', 'Yes', 'No, ask overwrite', 'Cancel', 'Yes');
    if strcmp(choice, 'Yes')
        idx = 2;
        while true
            candidate = sprintf('%s_%d', subject, idx);
            sp = fullfile(scheduleDir, [candidate cfg.paths.session2_schedule_suffix]);
            tp = fullfile(triggerDir, [candidate cfg.paths.session2_trigger_map_suffix]);
            if ~exist(sp, 'file') && ~exist(tp, 'file')
                outSubject = candidate; schedulePath = sp; triggerPath = tp; return;
            end
            idx = idx + 1;
        end
    elseif strcmp(choice, 'No, ask overwrite')
        choice2 = questdlg('Overwrite existing Session 2 schedule/trigger map?', 'Overwrite?', 'Yes', 'No', 'No');
        if strcmp(choice2, 'Yes')
            return;
        end
    end
    error('Aborted to avoid overwriting existing Session 2 schedule.');
end

function triggerMap = buildTriggerMap(schedule, cfg)
    n = height(schedule);
    triggerMap = table();
    triggerMap.subject_id = schedule.subject_id;
    triggerMap.session = schedule.session;
    triggerMap.trial = schedule.trial;
    triggerMap.block = schedule.block;
    triggerMap.image_filename = schedule.image_filename;
    triggerMap.old_new_status = schedule.old_new_status;
    triggerMap.image_onset_trigger = schedule.trigger_image_onset;
    triggerMap.recognition_yes_trigger = repmat(cfg.triggers.recognition_yes, n, 1);
    triggerMap.recognition_no_trigger = repmat(cfg.triggers.recognition_no, n, 1);
    triggerMap.kawaii_question_trigger = schedule.trigger_kawaii_question;
    triggerMap.kawaii_response_rating_1_to_9 = repmat("11-19", n, 1);
    triggerMap.arousal_question_trigger = schedule.trigger_arousal_question;
    triggerMap.arousal_response_rating_1_to_9 = repmat("21-29", n, 1);
    triggerMap.trial_reconstruction_note = repmat("trial_index_0_based = 10 * (arousal_question_trigger - 200) + (kawaii_question_trigger - 100)", n, 1);
end

function writeScheduleSummary(path, inputPath, schedulePath, triggerPath, schedule, cfg, seed, scheduleType, inputSubject)
    fid = fopen(path, 'w');
    if fid < 0; warning('Could not write summary: %s', path); return; end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, 'Session 2 recognition/rating schedule summary\n');
    fprintf(fid, '============================================\n');
    fprintf(fid, 'Input subject: %s\n', inputSubject);
    fprintf(fid, 'Schedule type: %s\n', scheduleType);
    fprintf(fid, 'Input pool: %s\n', inputPath);
    fprintf(fid, 'Schedule: %s\n', schedulePath);
    fprintf(fid, 'Trigger map: %s\n', triggerPath);
    fprintf(fid, 'Seed: %.0f\n', seed);
    fprintf(fid, 'Trials: %d\n', height(schedule));
    fprintf(fid, 'Blocks in generated schedule: %d\n', numel(unique(schedule.block)));
    blockIds = unique(schedule.block);
    blockCounts = zeros(numel(blockIds), 1);
    for bi = 1:numel(blockIds)
        blockCounts(bi) = sum(schedule.block == blockIds(bi));
    end
    fprintf(fid, 'Trials per block in generated schedule: %s\n', mat2str(blockCounts'));
    fprintf(fid, 'Available old images in Session 1 pool: %d\n', schedule.available_old_images(1));
    fprintf(fid, 'Available new images in Session 1 pool: %d\n', schedule.available_new_images(1));
    fprintf(fid, 'Configured maximum total trials: %d\n', schedule.max_total_trials_configured(1));
    fprintf(fid, 'Configured/planned number of blocks: %d\n', schedule.planned_n_blocks_configured(1));
    fprintf(fid, 'Planned maximum trials per block: %d\n', schedule.planned_trials_per_block(1));
    if strcmp(scheduleType, 'experiment') && height(schedule) < schedule.max_total_trials_configured(1)
        fprintf(fid, 'Adaptive note: generated fewer trials than the configured maximum because fewer valid old/new pairs were available.\n');
    end
    fprintf(fid, 'Old used: %d\n', sum(schedule.old_new_status == "old"));
    fprintf(fid, 'New used: %d\n', sum(schedule.old_new_status == "new"));
    fprintf(fid, 'Max old/new streak: %d\n', maxConsecutive(schedule.old_new_status));
    fprintf(fid, 'Fixation mode: %s\n', cfg.fixation.mode);
    fprintf(fid, 'Recognition timeout: %.3f s; rating timeout: %.3f s; timeouts enabled in runner config: %d\n', cfg.timeouts.recognition_s, cfg.timeouts.rating_s, cfg.timeouts.enabled);
end
