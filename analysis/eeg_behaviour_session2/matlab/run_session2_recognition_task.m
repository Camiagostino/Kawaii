function run_session2_recognition_task()
%RUN_SESSION2_RECOGNITION_TASK MATLAB/Psychtoolbox runner for Kawaii Session 2.
%
% Recognition/rating task:
%   image + yes/no recognition
%   same image + kawaii rating
%   same image + arousal rating
%
% This file is separate from Session 1 and does not modify the Session 1 task.

    clc;
    projectRoot = findProjectRoot();
    addpath(fullfile(projectRoot, 'config'));
    cfg = session2_recognition_config();
    cfg = normaliseSession2Config(cfg);

    opts = getRunOptions(cfg);
    if opts.cancelled; disp('Cancelled.'); return; end

    [schedulePath, scheduleSubject] = findSession2Schedule(projectRoot, cfg, opts.subject, opts.run_type);
    opts.schedule_subject = scheduleSubject;
    opts.subject = scheduleSubject;
    schedule = readtable(schedulePath, 'TextType', 'string');
    schedule = selectScheduleRows(schedule, opts);

    [outSubject, paths] = resolveLogPaths(projectRoot, cfg, opts.subject, opts.session2);
    opts.output_subject = outSubject;
    if ~strcmp(outSubject, opts.subject)
        schedule.subject_id(:) = string(outSubject);
    end

    fprintf('\nKawaii Session 2 recognition/rating task\n');
    fprintf('Subject schedule: %s\n', opts.subject);
    fprintf('Output/log ID: %s\n', opts.output_subject);
    fprintf('Run type: %s\n', opts.run_type);
    fprintf('Environment: %s\n', opts.environment);
    fprintf('Language: %s\n', opts.language);
    fprintf('Blocks selected: %s\n', mat2str(unique(schedule.block)'));
    fprintf('Trials selected: %d\n', height(schedule));
    fprintf('Schedule: %s\n', schedulePath);

    taskRows = struct([]); analysisRows = struct([]); timingRows = struct([]); triggerRows = struct([]); blockRows = struct([]);
    taskStatus = 'not_started'; abortReason = '';
    win = [];
    oldPriority = 0;
    eeg = initialiseEegIfNeeded(cfg, opts.environment);

    try
        AssertOpenGL;
        % Do not silently suppress Psychtoolbox sync/timing warnings.
        % For EEG, timing warnings need to be visible and investigated.
        skipSync = logical(getCfg(cfg.display, 'skip_sync_tests', false));
        if strcmp(opts.run_type, 'debug') && isfield(cfg.display, 'skip_sync_tests_debug')
            skipSync = logical(cfg.display.skip_sync_tests_debug);
        end
        Screen('Preference', 'SkipSyncTests', double(skipSync));
        if skipSync
            warning(['Psychtoolbox sync tests are being skipped by config. ' ...
                     'Use this only for local non-EEG troubleshooting, not real EEG data collection.']);
        end
        PsychDefaultSetup(2);
        screens = Screen('Screens');
        screenIndex = cfg.display.screen_index;
        if screenIndex < min(screens) || screenIndex > max(screens)
            screenIndex = max(screens);
        end
        useWindowedDebug = strcmp(opts.run_type, 'debug') && ...
            strcmp(opts.environment, 'behavioural') && logical(cfg.display.windowed_debug);
        if useWindowedDebug
            [win, rect] = PsychImaging('OpenWindow', screenIndex, cfg.display.background_colour, cfg.display.debug_window_rect);
        else
            [win, rect] = PsychImaging('OpenWindow', screenIndex, cfg.display.background_colour);
        end
        Screen('TextFont', win, 'Arial');
        Screen('TextSize', win, cfg.display.text_size);
        HideCursor;
        if cfg.display.use_priority
            oldPriority = Priority(MaxPriority(win));
        end
        ifi = Screen('GetFlipInterval', win);
        if getCfg(cfg.display, 'startup_flip_diagnostic', true)
            runStartupFlipDiagnostic(win, cfg, ifi);
        end

        text = languageText(cfg, opts.language);
        showTextScreen(win, cfg, text.instructions, cfg.keys.continue);

        selectedBlocks = unique(schedule.block)';
        nBlocksInSchedule = max(schedule.block);
        taskStatus = 'running';
        globalStart = GetSecs;
        for b = selectedBlocks
            blockFrame = schedule(schedule.block == b, :);
            blockStart = GetSecs;
            blockRows = appendStruct(blockRows, makeBlockRow(opts, b, height(blockFrame), blockStart - globalStart, 'started'));
            textures = preloadBlockTextures(win, projectRoot, blockFrame);

            for r = 1:height(blockFrame)
                row = blockFrame(r, :);
                trialResult = runOneSession2Trial(win, cfg, opts, text, eeg, projectRoot, row, textures, blockStart, globalStart);
                taskRows = appendStruct(taskRows, trialResult.raw);
                analysisRows = appendStruct(analysisRows, trialResult.analysis);
                timingRows = appendStruct(timingRows, trialResult.timing);
                triggerRows = appendStruct(triggerRows, trialResult.triggers);
            end
            releaseTextures(textures);
            blockEnd = GetSecs;
            blockRows = appendStruct(blockRows, makeBlockRow(opts, b, height(blockFrame), blockEnd - globalStart, 'finished'));

            if b ~= selectedBlocks(end)
                breakText = replaceTokens(text.break_screen, b, nBlocksInSchedule);
                showTextScreen(win, cfg, breakText, cfg.keys.continue);
            end
        end
        showTextScreen(win, cfg, text.ending, {}, 3.0);
        taskStatus = 'finished';

    catch ME
        taskStatus = 'aborted';
        abortReason = ME.message;
        fprintf(2, '\nERROR/ABORT: %s\n', ME.message);
        try
            fprintf(2, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        catch
        end
    end

    try
        if ~isempty(win); Screen('CloseAll'); end
        ShowCursor;
        if oldPriority > 0; Priority(oldPriority); end
    catch
    end

    writeSession2Logs(paths, opts, cfg, schedulePath, taskRows, analysisRows, timingRows, triggerRows, blockRows, taskStatus, abortReason);
    if strcmp(taskStatus, 'aborted')
        error('Task aborted: %s', abortReason);
    else
        fprintf('\nTask finished. Logs written to:\n%s\n', paths.baseDir);
    end
end

%% ========================================================================
%  MAIN TRIAL LOGIC
%  ========================================================================
function result = runOneSession2Trial(win, cfg, opts, text, eeg, projectRoot, row, textures, blockStart, globalStart)
    result.raw = struct(); result.analysis = struct(); result.timing = struct(); result.triggers = struct([]);
    trialStart = GetSecs;
    imageKey = char(string(row.image_path));
    if isempty(imageKey) || strcmpi(imageKey, 'nan'); imageKey = char(string(row.image_filename)); end
    tex = textures.(safeField(imageKey));

    fixationOnset = NaN; recognitionOnset = NaN; kawaiiOnset = NaN; arousalOnset = NaN;
    fixationMissed = NaN; recognitionMissed = NaN; kawaiiMissed = NaN; arousalMissed = NaN;
    fixationTimingWarning = ''; recognitionTimingWarning = ''; kawaiiTimingWarning = ''; arousalTimingWarning = '';
    recognitionTriggerTime = NaN; kawaiiQuestionTriggerTime = NaN; arousalQuestionTriggerTime = NaN;

    if cfg.fixation.enabled && row.fixation_duration_s > 0
        drawFixation(win, cfg);
        [fixationOnset, fixationMissed, fixationTimingWarning] = flipAndCheck(win, cfg, 'fixation');
        WaitSecs(row.fixation_duration_s);
    end

    drawRecognitionScreen(win, cfg, text, tex);
    [recognitionOnset, recognitionMissed, recognitionTimingWarning] = flipAndCheck(win, cfg, 'recognition_onset');
    [triggerRows1, recognitionTriggerTime] = sendTrigger(eeg, row.trigger_image_onset, 'image_recognition_onset', row, recognitionOnset, blockStart, globalStart);
    [recResp, recKey, recRt, recTimeout] = waitForLeftRightResponse(cfg, cfg.timeouts.recognition_s);
    recCode = NaN; recTrigTime = NaN; triggerRows2 = struct([]);
    if ~recTimeout
        if strcmp(recResp, 'yes'); recCode = cfg.triggers.recognition_yes; else; recCode = cfg.triggers.recognition_no; end
        [triggerRows2, recTrigTime] = sendTrigger(eeg, recCode, ['recognition_response_' recResp], row, GetSecs, blockStart, globalStart);
        showRecognitionSelectionFeedback(win, cfg, text, tex, recResp);
    end

    drawRatingScreen(win, cfg, text, tex, 'kawaii', cfg.rating.start_value);
    [kawaiiOnset, kawaiiMissed, kawaiiTimingWarning] = flipAndCheck(win, cfg, 'kawaii_question_onset');
    [triggerRows3, kawaiiQuestionTriggerTime] = sendTrigger(eeg, row.trigger_kawaii_question, 'kawaii_question_onset', row, kawaiiOnset, blockStart, globalStart);
    [kawaiiRating, kawaiiRt, kawaiiTimeout, kawaiiMoves] = waitForRating(win, cfg, text, tex, 'kawaii', cfg.timeouts.rating_s);
    kawaiiCode = NaN; kawaiiRespTrigTime = NaN; triggerRows4 = struct([]);
    if ~kawaiiTimeout
        kawaiiCode = cfg.triggers.kawaii_response_base + kawaiiRating;
        [triggerRows4, kawaiiRespTrigTime] = sendTrigger(eeg, kawaiiCode, 'kawaii_rating_response', row, GetSecs, blockStart, globalStart);
        showRatingSelectionFeedback(win, cfg, text, tex, 'kawaii', kawaiiRating);
    end

    drawRatingScreen(win, cfg, text, tex, 'arousal', cfg.rating.start_value);
    [arousalOnset, arousalMissed, arousalTimingWarning] = flipAndCheck(win, cfg, 'arousal_question_onset');
    [triggerRows5, arousalQuestionTriggerTime] = sendTrigger(eeg, row.trigger_arousal_question, 'arousal_question_onset', row, arousalOnset, blockStart, globalStart);
    [arousalRating, arousalRt, arousalTimeout, arousalMoves] = waitForRating(win, cfg, text, tex, 'arousal', cfg.timeouts.rating_s);
    arousalCode = NaN; arousalRespTrigTime = NaN; triggerRows6 = struct([]);
    if ~arousalTimeout
        arousalCode = cfg.triggers.arousal_response_base + arousalRating;
        [triggerRows6, arousalRespTrigTime] = sendTrigger(eeg, arousalCode, 'arousal_rating_response', row, GetSecs, blockStart, globalStart);
        showRatingSelectionFeedback(win, cfg, text, tex, 'arousal', arousalRating);
    end
    trialEnd = GetSecs;

    correctResponse = char(string(row.correct_recognition_response));
    recCorrect = ~recTimeout && strcmp(recResp, correctResponse);

    base = baseTrialStruct(opts, row);
    result.raw = base;
    result.raw.image_path_resolved = resolveImagePath(projectRoot, row);
    result.raw.trial_start_time = trialStart;
    result.raw.fixation_onset_time = fixationOnset;
    result.raw.recognition_onset_time = recognitionOnset;
    result.raw.kawaii_onset_time = kawaiiOnset;
    result.raw.arousal_onset_time = arousalOnset;
    result.raw.fixation_missed_flip_s = fixationMissed;
    result.raw.recognition_missed_flip_s = recognitionMissed;
    result.raw.kawaii_missed_flip_s = kawaiiMissed;
    result.raw.arousal_missed_flip_s = arousalMissed;
    result.raw.fixation_timing_warning = fixationTimingWarning;
    result.raw.recognition_timing_warning = recognitionTimingWarning;
    result.raw.kawaii_timing_warning = kawaiiTimingWarning;
    result.raw.arousal_timing_warning = arousalTimingWarning;
    result.raw.trial_end_time = trialEnd;
    result.raw.recognition_response = recResp;
    result.raw.recognition_key = recKey;
    result.raw.recognition_correct = recCorrect;
    result.raw.recognition_rt_s = recRt;
    result.raw.recognition_timeout = recTimeout;
    result.raw.kawaii_rating = kawaiiRating;
    result.raw.kawaii_rt_s = kawaiiRt;
    result.raw.kawaii_timeout = kawaiiTimeout;
    result.raw.kawaii_move_count = kawaiiMoves;
    result.raw.arousal_rating = arousalRating;
    result.raw.arousal_rt_s = arousalRt;
    result.raw.arousal_timeout = arousalTimeout;
    result.raw.arousal_move_count = arousalMoves;
    result.raw.image_trigger_time = recognitionTriggerTime;
    result.raw.kawaii_question_trigger_time = kawaiiQuestionTriggerTime;
    result.raw.arousal_question_trigger_time = arousalQuestionTriggerTime;
    result.raw.recognition_response_trigger_time = recTrigTime;
    result.raw.kawaii_response_trigger_time = kawaiiRespTrigTime;
    result.raw.arousal_response_trigger_time = arousalRespTrigTime;

    result.analysis = makeAnalysisRow(opts, row, recognitionOnset, kawaiiOnset, arousalOnset, trialEnd, blockStart, globalStart, ...
        recResp, recCorrect, recRt, recTimeout, kawaiiRating, kawaiiRt, kawaiiTimeout, arousalRating, arousalRt, arousalTimeout, ...
        recCode, kawaiiCode, arousalCode, fixationMissed, recognitionMissed, kawaiiMissed, arousalMissed, ...
        fixationTimingWarning, recognitionTimingWarning, kawaiiTimingWarning, arousalTimingWarning);
    result.timing = makeTimingRow(opts, row, blockStart, globalStart, fixationOnset, recognitionOnset, kawaiiOnset, arousalOnset, trialEnd, ...
        fixationMissed, recognitionMissed, kawaiiMissed, arousalMissed, ...
        fixationTimingWarning, recognitionTimingWarning, kawaiiTimingWarning, arousalTimingWarning);
    result.triggers = [triggerRows1 triggerRows2 triggerRows3 triggerRows4 triggerRows5 triggerRows6];
end

%% ========================================================================
%  DISPLAY FUNCTIONS
%  ========================================================================
function drawRecognitionScreen(win, cfg, text, tex, selectedResponse)
    if nargin < 5; selectedResponse = ''; end
    drawBaseImage(win, cfg, tex);
    Screen('TextSize', win, cfg.display.question_text_size);
    DrawFormattedText(win, text.recognition_question, 'center', yFrac(win, 0.67), cfg.display.text_colour);
    drawRecognitionLabel(win, cfg, text.recognition_left_label, xFrac(win, 0.38), yFrac(win, 0.78), strcmp(selectedResponse, 'yes'));
    drawRecognitionLabel(win, cfg, text.recognition_right_label, xFrac(win, 0.58), yFrac(win, 0.78), strcmp(selectedResponse, 'no'));
end

function drawRecognitionLabel(win, cfg, label, x, y, selected)
    baseSize = cfg.display.question_text_size;
    if selected
        Screen('TextSize', win, round(baseSize * getCfg(cfg.selection_feedback, 'recognition_scale_factor', 1.20)));
    else
        Screen('TextSize', win, baseSize);
    end
    DrawFormattedText(win, char(label), x, y, cfg.display.text_colour);
    Screen('TextSize', win, baseSize);
end

function showRecognitionSelectionFeedback(win, cfg, text, tex, selectedResponse)
    if ~isfield(cfg, 'selection_feedback') || ~getCfg(cfg.selection_feedback, 'enabled', true); return; end
    duration = getCfg(cfg.selection_feedback, 'duration_s', 0.5);
    if duration <= 0; return; end
    drawRecognitionScreen(win, cfg, text, tex, selectedResponse);
    Screen('Flip', win);
    WaitSecs(duration);
    postDelay = getCfg(cfg.timing, 'post_selection_feedback_delay_s', 0.0);
    if postDelay > 0; WaitSecs(postDelay); end
end

function drawRatingScreen(win, cfg, text, tex, ratingType, cursorValue, emphasiseSelected)
    if nargin < 7; emphasiseSelected = false; end
    drawBaseImage(win, cfg, tex);
    Screen('TextSize', win, cfg.display.question_text_size);
    if strcmp(ratingType, 'kawaii')
        question = text.kawaii_question; anchor = text.kawaii_anchor;
    else
        question = text.arousal_question; anchor = text.arousal_anchor;
    end
    DrawFormattedText(win, question, 'center', yFrac(win, 0.63), cfg.display.text_colour);
    Screen('TextSize', win, cfg.display.scale_text_size);
    DrawFormattedText(win, anchor, 'center', yFrac(win, 0.69), cfg.display.text_colour);
    drawScale(win, cfg, cursorValue, emphasiseSelected);
end

function showRatingSelectionFeedback(win, cfg, text, tex, ratingType, ratingValue)
    if ~isfield(cfg, 'selection_feedback') || ~getCfg(cfg.selection_feedback, 'enabled', true); return; end
    duration = getCfg(cfg.selection_feedback, 'duration_s', 0.5);
    if duration <= 0 || isnan(ratingValue); return; end
    drawRatingScreen(win, cfg, text, tex, ratingType, ratingValue, true);
    Screen('Flip', win);
    WaitSecs(duration);
    postDelay = getCfg(cfg.timing, 'post_selection_feedback_delay_s', 0.0);
    if postDelay > 0; WaitSecs(postDelay); end
end

function drawBaseImage(win, cfg, tex)
    Screen('FillRect', win, cfg.display.background_colour);
    [w, h] = Screen('WindowSize', win);
    side = cfg.display.image_size_fraction(1) * h;
    dst = CenterRectOnPointd([0 0 side side], w/2, yFrac(win, cfg.display.image_y_fraction));
    Screen('DrawTexture', win, tex, [], dst);
end

function drawScale(win, cfg, cursorValue, emphasiseSelected)
    if nargin < 4; emphasiseSelected = false; end
    [w, h] = Screen('WindowSize', win);
    values = cfg.rating.min_value:cfg.rating.max_value;
    xStart = 0.30 * w; xEnd = 0.70 * w; y = 0.78 * h;
    xs = linspace(xStart, xEnd, numel(values));
    baseSize = cfg.display.scale_text_size;
    for i = 1:numel(values)
        selected = values(i) == cursorValue;
        if selected && emphasiseSelected
            Screen('TextSize', win, round(baseSize * getCfg(cfg.selection_feedback, 'rating_scale_factor', 1.20)));
        else
            Screen('TextSize', win, baseSize);
        end
        label = num2str(values(i));
        bounds = Screen('TextBounds', win, label);
        textWidth = bounds(3) - bounds(1);
        textHeight = bounds(4) - bounds(2);
        x = xs(i) - textWidth/2;
        DrawFormattedText(win, label, x, y, cfg.display.text_colour);
        if selected && cfg.rating.show_cursor_box
            padX = getCfg(cfg.rating, 'cursor_box_x_padding_px', 10);
            padY = getCfg(cfg.rating, 'cursor_box_y_padding_px', 8);
            xOffset = getCfg(cfg.rating, 'cursor_box_x_offset_px', 0);
            yOffset = getCfg(cfg.rating, 'cursor_box_y_offset_px', -17);
            rect = [x - padX + xOffset, y - padY + yOffset, x + textWidth + padX + xOffset, y + textHeight + padY + yOffset];
            Screen('FrameRect', win, cfg.display.text_colour, rect, cfg.rating.cursor_line_width);
        end
    end
    Screen('TextSize', win, baseSize);
end

function drawFixation(win, cfg)
    Screen('FillRect', win, cfg.display.background_colour);
    Screen('TextSize', win, cfg.display.text_size);
    DrawFormattedText(win, cfg.fixation.text, 'center', 'center', cfg.display.text_colour);
end

function [onsetTime, missedFlip, warningText] = flipAndCheck(win, cfg, phaseName)
    % Flip the display and keep Psychtoolbox's missed-deadline estimate.
    % For response-paced screens the flip is immediate, but keeping this value
    % in the logs makes timing QC explicit and keeps EEG-relevant events tied
    % to actual visual-onset timestamps.
    [onsetTime, ~, ~, missedFlip] = Screen('Flip', win);
    if isempty(missedFlip) || ~isnumeric(missedFlip) || isnan(missedFlip)
        missedFlip = 0;
    end
    threshold = getCfg(cfg.display, 'timing_warning_threshold_s', 0.020);
    warningText = '';
    if missedFlip > threshold
        warningText = sprintf('%s missed flip/deadline by %.6f s', char(phaseName), missedFlip);
        warning('%s', warningText);
    end
end

function showTextScreen(win, cfg, text, waitKeys, fixedDuration)
    if nargin < 5; fixedDuration = []; end
    Screen('FillRect', win, cfg.display.background_colour);
    Screen('TextSize', win, cfg.display.text_size);
    DrawFormattedText(win, char(text), 'center', 'center', cfg.display.text_colour, 80, [], [], 1.3);
    Screen('Flip', win);
    if ~isempty(fixedDuration) && fixedDuration > 0
        WaitSecs(fixedDuration);
    end
    if nargin >= 4 && ~isempty(waitKeys)
        waitForKeys(waitKeys, cfg.keys.quit);
    end
end

function runStartupFlipDiagnostic(win, cfg, ifi)
    % Lightweight flip stability check, inspired by the lab RI-SIGA pattern of
    % using actual Screen('Flip') timestamps. This does not hide PTB sync
    % warnings; it gives an extra logged/printed clue after the window opens.
    nFlips = round(getCfg(cfg.display, 'startup_flip_diagnostic_flips', 30));
    if nFlips <= 1; return; end
    warningThreshold = getCfg(cfg.display, 'startup_flip_warning_threshold_s', 0.002);
    Screen('FillRect', win, cfg.display.background_colour);
    vbl = Screen('Flip', win);
    deltas = nan(nFlips, 1);
    for i = 1:nFlips
        Screen('FillRect', win, cfg.display.background_colour);
        vbl2 = Screen('Flip', win, vbl + 0.5 * ifi);
        deltas(i) = abs((vbl2 - vbl) - ifi);
        vbl = vbl2;
    end
    maxDev = max(deltas);
    if maxDev > warningThreshold
        warning('Startup flip diagnostic: max frame-interval deviation %.4f s exceeds %.4f s. Check display/sync setup, especially before EEG.', maxDev, warningThreshold);
    else
        fprintf('Startup flip diagnostic OK: max frame-interval deviation %.4f s.\n', maxDev);
    end
end

%% ========================================================================
%  RESPONSE FUNCTIONS
%  ========================================================================
function [resp, keyName, rt, timedOut] = waitForLeftRightResponse(cfg, timeoutS)
    if ~cfg.timeouts.enabled; timeoutS = Inf; end
    start = GetSecs; resp = ''; keyName = ''; rt = NaN; timedOut = false;
    KbReleaseWait;
    while true
        [down, secs, keyCode] = KbCheck;
        if down
            names = firstKeyName(keyCode);
            if strcmp(names, lower(cfg.keys.quit)); error('Aborted by user.'); end
            if strcmp(names, lower(cfg.keys.left))
                resp = 'yes'; keyName = names; rt = secs - start; KbReleaseWait; return;
            elseif strcmp(names, lower(cfg.keys.right))
                resp = 'no'; keyName = names; rt = secs - start; KbReleaseWait; return;
            end
        end
        if GetSecs - start >= timeoutS
            timedOut = true; rt = NaN; return;
        end
        WaitSecs(0.002);
    end
end

function [rating, rt, timedOut, moves] = waitForRating(win, cfg, text, tex, ratingType, timeoutS)
    if ~cfg.timeouts.enabled; timeoutS = Inf; end
    start = GetSecs; rating = cfg.rating.start_value; rt = NaN; timedOut = false; moves = 0;
    KbReleaseWait;
    while true
        [down, secs, keyCode] = KbCheck;
        if down
            names = firstKeyName(keyCode);
            if strcmp(names, lower(cfg.keys.quit)); error('Aborted by user.'); end
            if strcmp(names, lower(cfg.keys.rating_left))
                rating = max(cfg.rating.min_value, rating - 1); moves = moves + 1;
                drawRatingScreen(win, cfg, text, tex, ratingType, rating); Screen('Flip', win); KbReleaseWait;
            elseif strcmp(names, lower(cfg.keys.rating_right))
                rating = min(cfg.rating.max_value, rating + 1); moves = moves + 1;
                drawRatingScreen(win, cfg, text, tex, ratingType, rating); Screen('Flip', win); KbReleaseWait;
            elseif strcmp(names, lower(cfg.keys.rating_confirm))
                rt = secs - start; KbReleaseWait; return;
            end
        end
        if GetSecs - start >= timeoutS
            timedOut = true; rt = NaN; rating = NaN; return;
        end
        WaitSecs(0.002);
    end
end

function waitForKeys(keys, quitKey)
    if ischar(keys) || isstring(keys); keys = cellstr(string(keys)); end
    keys = lower(string(keys)); quitKey = lower(string(quitKey));
    KbReleaseWait;
    while true
        [down, ~, keyCode] = KbCheck;
        if down
            name = string(firstKeyName(keyCode));
            if name == quitKey; error('Aborted by user.'); end
            if any(name == keys); KbReleaseWait; return; end
        end
        WaitSecs(0.01);
    end
end

%% ========================================================================
%  TRIGGERS
%  ========================================================================
function eeg = initialiseEegIfNeeded(cfg, environment)
    eeg.enabled = strcmp(environment, 'eeg') && cfg.eeg.send_triggers;
    eeg.ioObj = [];
    eeg.address = [];
    eeg.cfg = cfg;
    if ~eeg.enabled; return; end
    try
        global cogent
        config_io;
       % cogent.io.ioObj = io64();
        eeg.ioObj = cogent.io.ioObj;
        eeg.address = hex2dec(cfg.eeg.port_address_hex);
        io64(eeg.ioObj, eeg.address, cfg.eeg.reset_code);
        fprintf('EEG triggers enabled at address 0x%s.\n', cfg.eeg.port_address_hex);
    catch ME
        error('Could not initialise EEG trigger interface: %s', ME.message);
    end
end

function [triggerRows, trigTime] = sendTrigger(eeg, code, eventName, schedRow, eventTime, blockStart, globalStart)
    triggerRows = struct([]); trigTime = NaN;
    if isnan(code) || isempty(code); return; end
    if eeg.enabled
        io64(eeg.ioObj, eeg.address, double(code));
        trigTime = GetSecs;
        WaitSecs(eeg.cfg.eeg.pulse_width_s);
        io64(eeg.ioObj, eeg.address, eeg.cfg.eeg.reset_code);
    else
        trigTime = GetSecs;
    end
    triggerRows = struct();
    triggerRows.subject_id = char(string(schedRow.subject_id));
    triggerRows.session = double(schedRow.session);
    triggerRows.block = double(schedRow.block);
    triggerRows.trial = double(schedRow.trial);
    triggerRows.event_name = char(eventName);
    triggerRows.trigger_code = double(code);
    triggerRows.event_time = eventTime;
    triggerRows.trigger_time = trigTime;
    triggerRows.event_onset_block_s = eventTime - blockStart;
    triggerRows.event_onset_session_s = eventTime - globalStart;
    triggerRows.trigger_delay_s = trigTime - eventTime;
end

%% ========================================================================
%  LOGGING
%  ========================================================================
function base = baseTrialStruct(opts, row)
    base = struct();
    base.subject_id = opts.output_subject;
    base.session = double(row.session);
    base.mode = [opts.run_type '/' opts.environment];
    base.language = opts.language;
    base.block = double(row.block);
    base.trial = double(row.trial);
    base.trial_in_block = double(row.trial_in_block);
    base.image_filename = char(string(row.image_filename));
    base.image_path = char(string(row.image_path));
    base.old_new_status = char(string(row.old_new_status));
    base.correct_recognition_response = char(string(row.correct_recognition_response));
    base.trial_digit_0_9 = double(row.trial_digit_0_9);
    base.trial_group_0_39 = double(row.trial_group_0_39);
end

function analysis = makeAnalysisRow(opts, row, recOnset, kawaiiOnset, arousalOnset, trialEnd, blockStart, globalStart, recResp, recCorrect, recRt, recTimeout, kawaiiRating, kawaiiRt, kawaiiTimeout, arousalRating, arousalRt, arousalTimeout, recCode, kawaiiCode, arousalCode, fixationMissed, recognitionMissed, kawaiiMissed, arousalMissed, fixationWarning, recognitionWarning, kawaiiWarning, arousalWarning)
    analysis = struct();
    analysis.subject_id = opts.output_subject;
    analysis.session = double(row.session);
    analysis.mode = [opts.run_type '/' opts.environment];
    analysis.language = opts.language;
    analysis.block = double(row.block);
    analysis.trial = double(row.trial);
    analysis.trial_in_block = double(row.trial_in_block);
    analysis.image_filename = char(string(row.image_filename));
    analysis.image_path = char(string(row.image_path));
    analysis.old_new_status = char(string(row.old_new_status));
    analysis.correct_recognition_response = char(string(row.correct_recognition_response));
    analysis.recognition_response = recResp;
    analysis.recognition_correct = recCorrect;
    analysis.recognition_rt_s = recRt;
    analysis.recognition_timeout = recTimeout;
    analysis.kawaii_rating = kawaiiRating;
    analysis.kawaii_rt_s = kawaiiRt;
    analysis.kawaii_timeout = kawaiiTimeout;
    analysis.arousal_rating = arousalRating;
    analysis.arousal_rt_s = arousalRt;
    analysis.arousal_timeout = arousalTimeout;
    analysis.recognition_onset_s = recOnset - globalStart;
    analysis.kawaii_onset_s = kawaiiOnset - globalStart;
    analysis.arousal_onset_s = arousalOnset - globalStart;
    analysis.trial_end_s = trialEnd - globalStart;
    % Extra block-relative columns are kept for diagnostics only.
    analysis.recognition_onset_block_s = recOnset - blockStart;
    analysis.kawaii_onset_block_s = kawaiiOnset - blockStart;
    analysis.arousal_onset_block_s = arousalOnset - blockStart;
    analysis.trial_end_block_s = trialEnd - blockStart;
    analysis.image_onset_trigger = double(row.trigger_image_onset);
    analysis.recognition_response_trigger = recCode;
    analysis.kawaii_onset_trigger = double(row.trigger_kawaii_question);
    analysis.kawaii_response_trigger = kawaiiCode;
    analysis.arousal_onset_trigger = double(row.trigger_arousal_question);
    analysis.arousal_response_trigger = arousalCode;
    analysis.recognition_missed_flip_s = recognitionMissed;
    analysis.kawaii_missed_flip_s = kawaiiMissed;
    analysis.arousal_missed_flip_s = arousalMissed;
    analysis.any_missed_flip = any([fixationMissed recognitionMissed kawaiiMissed arousalMissed] > 0);
    analysis.timing_warning = combineTimingWarnings({fixationWarning, recognitionWarning, kawaiiWarning, arousalWarning});
    analysis.fallback_used = false;
    optional = {'M_Kawaii','M_Arousal','species','mammal_status','animal_class','source_feedback_type','source_run','source_trial'};
    for i = 1:numel(optional)
        col = optional{i};
        if ismember(col, row.Properties.VariableNames)
            analysis.(col) = row.(col);
        else
            analysis.(col) = '';
        end
    end
end

function timing = makeTimingRow(opts, row, blockStart, globalStart, fixationOnset, recOnset, kawaiiOnset, arousalOnset, trialEnd, fixationMissed, recognitionMissed, kawaiiMissed, arousalMissed, fixationWarning, recognitionWarning, kawaiiWarning, arousalWarning)
    timing = baseTrialStruct(opts, row);
    timing.block_start_time = blockStart;
    timing.task_start_time = globalStart;
    timing.fixation_onset_time = fixationOnset;
    timing.recognition_onset_time = recOnset;
    timing.kawaii_onset_time = kawaiiOnset;
    timing.arousal_onset_time = arousalOnset;
    timing.trial_end_time = trialEnd;
    timing.recognition_onset_s = recOnset - globalStart;
    timing.kawaii_onset_s = kawaiiOnset - globalStart;
    timing.arousal_onset_s = arousalOnset - globalStart;
    timing.trial_end_s = trialEnd - globalStart;
    timing.recognition_onset_block_s = recOnset - blockStart;
    timing.kawaii_onset_block_s = kawaiiOnset - blockStart;
    timing.arousal_onset_block_s = arousalOnset - blockStart;
    timing.trial_end_block_s = trialEnd - blockStart;
    timing.fixation_missed_flip_s = fixationMissed;
    timing.recognition_missed_flip_s = recognitionMissed;
    timing.kawaii_missed_flip_s = kawaiiMissed;
    timing.arousal_missed_flip_s = arousalMissed;
    timing.any_missed_flip = any([fixationMissed recognitionMissed kawaiiMissed arousalMissed] > 0);
    timing.timing_warning = combineTimingWarnings({fixationWarning, recognitionWarning, kawaiiWarning, arousalWarning});
end

function row = makeBlockRow(opts, block, nTrials, onsetS, status)
    row = struct();
    row.subject_id = opts.output_subject;
    row.session = opts.session2;
    row.mode = [opts.run_type '/' opts.environment];
    row.language = opts.language;
    row.block = block;
    row.n_trials = nTrials;
    row.onset_s = onsetS;
    row.status = status;
end

function writeSession2Logs(paths, opts, cfg, schedulePath, taskRows, analysisRows, timingRows, triggerRows, blockRows, taskStatus, abortReason)
    if ~exist(paths.baseDir, 'dir'); mkdir(paths.baseDir); end

    % Use a robust writer rather than direct struct2table(...). Some MATLAB
    % values returned from tables/keyboard routines can be small arrays or
    % strings. Direct struct2table can crash if one field has a different
    % number of rows, which is especially bad at block/end transitions.
    writeStructLog(taskRows, paths.rawLog);
    writeStructLog(analysisRows, paths.analysisLog);
    writeStructLog(timingRows, paths.timingLog);
    writeStructLog(triggerRows, paths.triggerLog);
    writeStructLog(blockRows, paths.blockLog);

    status = table(string(opts.output_subject), opts.session2, string(opts.run_type), string(opts.environment), string(opts.language), string(taskStatus), string(abortReason), string(schedulePath), ...
        'VariableNames', {'subject_id','session','run_type','environment','language','task_status','abort_reason','schedule_file'});
    writetable(status, paths.statusLog);
    if cfg.outputs.save_config_copy
        copyfile(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config', 'session2_recognition_config.m'), paths.configCopy);
    end
end

function writeStructLog(rows, filePath)
    if isempty(rows)
        writetable(table(), filePath);
        return;
    end
    try
        writetable(safeStructArrayToTable(rows), filePath);
    catch ME
        warning('Could not write %s with safe table writer (%s). Writing fallback status table instead.', filePath, ME.message);
        fallback = table(string(ME.message), 'VariableNames', {'log_write_error'});
        writetable(fallback, filePath);
    end
end

function tbl = safeStructArrayToTable(rows)
    % Convert a struct or struct array into a table where every field has one
    % value per row. Numeric/logical scalar columns are kept numeric; anything
    % complex/non-scalar is converted to a readable string so writetable cannot
    % fail because of mixed row sizes.
    if isempty(rows)
        tbl = table();
        return;
    end

    fields = fieldnames(rows);
    nRows = numel(rows);
    tbl = table();

    for f = 1:numel(fields)
        fieldName = fields{f};
        values = cell(nRows, 1);
        numericScalar = true;
        logicalScalar = true;

        for r = 1:nRows
            if isfield(rows(r), fieldName)
                value = rows(r).(fieldName);
            else
                value = '';
            end
            values{r} = scalariseLogValue(value);
            numericScalar = numericScalar && isnumeric(values{r}) && isscalar(values{r});
            logicalScalar = logicalScalar && islogical(values{r}) && isscalar(values{r});
        end

        if numericScalar
            col = nan(nRows, 1);
            for r = 1:nRows
                col(r) = double(values{r});
            end
            tbl.(fieldName) = col;
        elseif logicalScalar
            col = false(nRows, 1);
            for r = 1:nRows
                col(r) = logical(values{r});
            end
            tbl.(fieldName) = col;
        else
            col = strings(nRows, 1);
            for r = 1:nRows
                col(r) = string(values{r});
            end
            tbl.(fieldName) = col;
        end
    end
end

function value = scalariseLogValue(value)
    % Keep simple scalar values useful for analysis, but turn arrays/cells into
    % compact strings. This prevents struct2table-style crashes at log writing.
    try
        if istable(value)
            value = table2cell(value);
        end
        if iscell(value)
            if isempty(value)
                value = '';
            elseif numel(value) == 1
                value = scalariseLogValue(value{1});
            else
                parts = strings(1, numel(value));
                for i = 1:numel(value)
                    parts(i) = string(scalariseLogValue(value{i}));
                end
                value = strjoin(parts, ';');
            end
        elseif isstring(value)
            if isempty(value)
                value = '';
            elseif isscalar(value)
                value = char(value);
            else
                value = strjoin(value(:).', ';');
                value = char(value);
            end
        elseif ischar(value)
            % already fine
        elseif isnumeric(value) || islogical(value)
            if isempty(value)
                value = NaN;
            elseif ~isscalar(value)
                value = mat2str(value);
            end
        else
            value = char(string(value));
        end
    catch
        value = '<unwritable_log_value>';
    end
end

function [outSubject, paths] = resolveLogPaths(projectRoot, cfg, subject, session2)
    outSubject = subject;
    paths = makeLogPaths(projectRoot, cfg, outSubject, session2);
    expected = {paths.rawLog, paths.analysisLog, paths.timingLog, paths.triggerLog, paths.blockLog, paths.statusLog};
    existsAny = any(cellfun(@(p) exist(p, 'file') > 0, expected));
    if ~existsAny; return; end
    if cfg.outputs.offer_suffix_if_exists
        choice = questdlg(sprintf('Session 2 logs already exist for %s. Save as next suffix instead?', subject), 'Existing logs', 'Yes', 'No, ask overwrite', 'Cancel', 'Yes');
        if strcmp(choice, 'Yes')
            idx = 2;
            while true
                candidate = sprintf('%s_%d', subject, idx);
                candPaths = makeLogPaths(projectRoot, cfg, candidate, session2);
                candExpected = {candPaths.rawLog, candPaths.analysisLog, candPaths.timingLog, candPaths.triggerLog, candPaths.blockLog, candPaths.statusLog};
                if ~any(cellfun(@(p) exist(p, 'file') > 0, candExpected))
                    outSubject = candidate; paths = candPaths; return;
                end
                idx = idx + 1;
            end
        elseif strcmp(choice, 'No, ask overwrite') && cfg.outputs.allow_overwrite_after_decline_suffix
            choice2 = questdlg('Overwrite existing Session 2 logs?', 'Overwrite?', 'Yes', 'No', 'No');
            if strcmp(choice2, 'Yes'); return; end
        end
    end
    error('Aborted to avoid overwriting existing logs.');
end

function paths = makeLogPaths(projectRoot, cfg, subject, session2)
    base = sprintf('%s_session%d', subject, session2);
    paths.baseDir = fullfile(projectRoot, cfg.paths.session2_log_dir, base);
    paths.rawLog = fullfile(paths.baseDir, [base '_task_log_raw.csv']);
    paths.analysisLog = fullfile(paths.baseDir, [base '_trial_log_analysis.csv']);
    paths.timingLog = fullfile(paths.baseDir, [base '_timing_log.csv']);
    paths.triggerLog = fullfile(paths.baseDir, [base '_trigger_log.csv']);
    paths.blockLog = fullfile(paths.baseDir, [base '_block_log.csv']);
    paths.statusLog = fullfile(paths.baseDir, [base '_status.csv']);
    paths.configCopy = fullfile(paths.baseDir, [base '_config_used.m']);
end

%% ========================================================================
%  OPTIONS / SCHEDULE SELECTION
%  ========================================================================
function opts = getRunOptions(cfg)
    opts.cancelled = false;
    prompt = { ...
        'Subject ID (e.g. 1, sub001, pilot00):', ...
        'Run type: experiment/default (1) or debug (2):', ...
        'Environment: EEG/default (2) or behavioural (1):', ...
        'Language: English/default (1) or German (2):', ...
        'Block(s): blank/all, 1, 1 3, or 1-3:', ...
        'Debug trials per selected block (debug only; blank = all):'};
    defaults = {'1','1','2','1','',''};
    answer = inputdlg(prompt, 'Kawaii Session 2 recognition/rating task', [1 80], defaults);
    if isempty(answer); opts.cancelled = true; return; end
    opts.subject = char(normaliseSubjectId(answer{1}));
    opts.session2 = cfg.task.session2_id;
    opts.run_type = parseRunType(answer{2});
    opts.environment = parseEnvironment(answer{3});
    opts.language = parseLanguage(answer{4});
    opts.selected_blocks_raw = strtrim(answer{5});
    opts.debug_trials_per_block = str2double(strtrim(answer{6}));
    if isnan(opts.debug_trials_per_block); opts.debug_trials_per_block = []; end
    if strcmp(opts.run_type, 'experiment') && ~isempty(opts.debug_trials_per_block)
        error('Experiment mode does not allow shortened trial counts. Use debug mode for trial limits.');
    end
end

function schedule = selectScheduleRows(schedule, opts)
    blocks = parseBlocks(opts.selected_blocks_raw, unique(schedule.block)');
    schedule = schedule(ismember(schedule.block, blocks), :);
    if strcmp(opts.run_type, 'debug') && ~isempty(opts.debug_trials_per_block)
        keep = false(height(schedule), 1);
        for i = 1:numel(blocks)
            idx = find(schedule.block == blocks(i));
            keep(idx(1:min(numel(idx), opts.debug_trials_per_block))) = true;
        end
        schedule = schedule(keep, :);
    end
end

function blocks = parseBlocks(raw, allBlocks)
    raw = strtrim(char(raw));
    if isempty(raw) || strcmpi(raw, 'all')
        blocks = allBlocks; return;
    end
    blocks = [];
    parts = regexp(raw, '[,;\s]+', 'split');
    for i = 1:numel(parts)
        part = strtrim(parts{i}); if isempty(part); continue; end
        if contains(part, '-')
            lr = regexp(part, '-', 'split');
            blocks = [blocks str2double(lr{1}):str2double(lr{2})]; %#ok<AGROW>
        else
            blocks = [blocks str2double(part)]; %#ok<AGROW>
        end
    end
    blocks = unique(blocks(~isnan(blocks)));
    if isempty(blocks); error('No valid blocks selected.'); end
end

function runType = parseRunType(x)
    x = lower(strtrim(char(x)));
    if isempty(x) || strcmp(x, '1') || strcmp(x, 'experiment'); runType = 'experiment';
    elseif strcmp(x, '2') || strcmp(x, 'debug'); runType = 'debug';
    else; error('Invalid run type: %s', x); end
end

function env = parseEnvironment(x)
    x = lower(strtrim(char(x)));
    if isempty(x) || strcmp(x, '2') || strcmp(x, 'eeg'); env = 'eeg';
    elseif strcmp(x, '1') || strcmp(x, 'behavioural') || strcmp(x, 'behavioral'); env = 'behavioural';
    else; error('Invalid environment: %s', x); end
end

function lang = parseLanguage(x)
    x = lower(strtrim(char(x)));
    if isempty(x) || strcmp(x, '1') || strcmp(x, 'en') || strcmp(x, 'english'); lang = 'en';
    elseif strcmp(x, '2') || strcmp(x, 'de') || strcmp(x, 'german'); lang = 'de';
    else; error('Invalid language: %s', x); end
end

function subject = normaliseSubjectId(raw)
    raw = string(strtrim(char(raw)));
    if ~isempty(regexp(char(raw), '^\d+$', 'once'))
        subject = string(sprintf('sub%03d', str2double(raw)));
    else
        subject = raw;
    end
end

function text = languageText(cfg, lang)
    if strcmp(lang, 'de'); text = cfg.text.de; else; text = cfg.text.en; end
end

%% ========================================================================
%  IMAGE PRELOADING / UTILS
%  ========================================================================
function textures = preloadBlockTextures(win, projectRoot, blockFrame)
    textures = struct();
    for i = 1:height(blockFrame)
        key = char(string(blockFrame.image_path(i)));
        if isempty(key) || strcmpi(key, 'nan'); key = char(string(blockFrame.image_filename(i))); end
        field = safeField(key);
        if isfield(textures, field); continue; end
        imgPath = resolveImagePath(projectRoot, blockFrame(i, :));
        if ~exist(imgPath, 'file')
            error('Image not found: %s', imgPath);
        end
        img = imread(imgPath);
        textures.(field) = Screen('MakeTexture', win, img);
    end
end

function releaseTextures(textures)
    if isempty(textures); return; end
    f = fieldnames(textures);
    for i = 1:numel(f)
        try Screen('Close', textures.(f{i})); catch; end %#ok<CTCH>
    end
end

function path = resolveImagePath(projectRoot, row)
    p = char(string(row.image_path));
    if isempty(p) || strcmpi(p, 'nan')
        p = char(string(row.image_filename));
    end
    if exist(p, 'file')
        path = p;
    else
        path = fullfile(projectRoot, p);
    end
end

function field = safeField(key)
    field = matlab.lang.makeValidName(['img_' char(javaStringHash(key))]);
end

function h = javaStringHash(s)
    % Simple deterministic hash that does not require Java.
    bytes = double(char(s));
    hval = uint32(2166136261);
    for i = 1:numel(bytes)
        hval = bitxor(hval, uint32(bytes(i)));
        hval = uint32(mod(double(hval) * 16777619, 2^32));
    end
    h = sprintf('%08x', hval);
end

function s = replaceTokens(s, block, nBlocks)
    s = char(s);
    s = strrep(s, '{block}', num2str(block));
    s = strrep(s, '{n_blocks}', num2str(nBlocks));
end

function x = xFrac(win, frac)
    [w, ~] = Screen('WindowSize', win); x = frac * w;
end
function y = yFrac(win, frac)
    [~, h] = Screen('WindowSize', win); y = frac * h;
end

function value = getCfg(structure, fieldName, defaultValue)
    if isstruct(structure) && isfield(structure, fieldName)
        value = structure.(fieldName);
    else
        value = defaultValue;
    end
end

function text = combineTimingWarnings(parts)
    text = '';
    for i = 1:numel(parts)
        part = char(string(parts{i}));
        if ~isempty(strtrim(part))
            if isempty(text)
                text = part;
            else
                text = [text ' | ' part]; %#ok<AGROW>
            end
        end
    end
end

function out = appendStruct(out, item)
    if isempty(item)
        return;
    end
    if isempty(out)
        out = item;
    else
        % item may be a struct array, e.g. several trigger events from one trial.
        out = [out item]; %#ok<AGROW>
    end
end

function name = firstKeyName(keyCode)
    idx = find(keyCode, 1, 'first');
    if isempty(idx)
        name = '';
        return;
    end
    names = KbName(idx);
    if iscell(names); names = names{1}; end
    name = lower(char(names));
end

function [schedulePath, scheduleSubject] = findSession2Schedule(projectRoot, cfg, subject, runType)
    scheduleDir = fullfile(projectRoot, cfg.paths.session2_schedule_dir);
    suffix = cfg.paths.session2_schedule_suffix;
    candidates = {};
    if strcmp(runType, 'debug')
        if ~endsWith(subject, '_debug')
            candidates{end+1} = [subject '_debug']; %#ok<AGROW>
        end
        candidates{end+1} = subject; %#ok<AGROW>
    else
        candidates{end+1} = subject; %#ok<AGROW>
        if ~endsWith(subject, '_debug')
            candidates{end+1} = [subject '_debug']; %#ok<AGROW>
        end
    end
    tried = cell(size(candidates));
    for i = 1:numel(candidates)
        candidate = candidates{i};
        path = fullfile(scheduleDir, [candidate suffix]);
        tried{i} = path;
        if exist(path, 'file')
            schedulePath = path;
            scheduleSubject = candidate;
            return;
        end
    end
    error('Session 2 schedule not found. Tried:\n%s\n\nRun matlab/generate_session2_recognition_schedule.m first.', strjoin(tried, '\n'));
end

function cfg = normaliseSession2Config(cfg)
    % Accept colour names or numeric RGB triplets in the config. Psychtoolbox
    % FillRect requires numeric colours, so convert once immediately after load.
    cfg.display.background_colour = colourToRgb(cfg.display.background_colour, [0 0 0]);
    cfg.display.text_colour = colourToRgb(cfg.display.text_colour, [255 255 255]);
    if ~isfield(cfg, 'selection_feedback'); cfg.selection_feedback = struct(); end
    if ~isfield(cfg.selection_feedback, 'enabled'); cfg.selection_feedback.enabled = true; end
    if ~isfield(cfg.selection_feedback, 'duration_s'); cfg.selection_feedback.duration_s = 0.5; end
    if ~isfield(cfg.selection_feedback, 'recognition_scale_factor'); cfg.selection_feedback.recognition_scale_factor = 1.20; end
    if ~isfield(cfg.selection_feedback, 'rating_scale_factor'); cfg.selection_feedback.rating_scale_factor = 1.20; end
    if ~isfield(cfg, 'timing'); cfg.timing = struct(); end
    if ~isfield(cfg.timing, 'post_selection_feedback_delay_s'); cfg.timing.post_selection_feedback_delay_s = 0.0; end
    if ~isfield(cfg.rating, 'cursor_box_x_padding_px'); cfg.rating.cursor_box_x_padding_px = 10; end
    if ~isfield(cfg.rating, 'cursor_box_y_padding_px'); cfg.rating.cursor_box_y_padding_px = 8; end
    if ~isfield(cfg.rating, 'cursor_box_x_offset_px'); cfg.rating.cursor_box_x_offset_px = 0; end
    if ~isfield(cfg.rating, 'cursor_box_y_offset_px'); cfg.rating.cursor_box_y_offset_px = -17; end
end

function rgb = colourToRgb(value, defaultRgb)
    rgb = defaultRgb;
    try
        if isnumeric(value)
            rgb = double(value);
            if numel(rgb) == 1; rgb = [rgb rgb rgb]; end
            rgb = reshape(rgb, 1, []);
            return;
        end
        name = lower(strtrim(char(string(value))));
        switch name
            case {'black','k'}
                rgb = [0 0 0];
            case {'white','w'}
                rgb = [255 255 255];
            case {'grey','gray'}
                rgb = [128 128 128];
            otherwise
                rgb = defaultRgb;
        end
    catch
        rgb = defaultRgb;
    end
end

function projectRoot = findProjectRoot()
    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(here);
    if ~exist(fullfile(projectRoot, 'config'), 'dir')
        error('Could not find project root from MATLAB script location.');
    end
end
