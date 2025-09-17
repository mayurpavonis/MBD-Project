%% run_preprocessmat.m

% -------------------------
% 1) Load data (example)
% -------------------------
% If your workspace already contains gx_noisy, gx_GT, time, skip the load.
% if ~exist('gx_noisy','var') || ~exist('gx_GT','var') || ~exist('time','var')
%     % Change filename and variable names to match your file
%     s = load('your_data.mat'); 
%     gx_noisy = s.gx_noisy;
%     gx_GT    = s.gx_GT;
%     if isfield(s,'time'), time = s.time; end
% end

% -------------------------
% 2) Optional: build from windows (if you have windows)
% -------------------------
% Example: wins_noisy is LxW (each column one window). Uncomment only if needed.
% if exist('wins_noisy','var') && exist('wins_gt','var')
%     % ensure windows are columns: L x W
%     noisy_vec = reshape(wins_noisy, [], 1);   % stack columns in order
%     gt_vec    = reshape(wins_gt,    [], 1);
%     % create time if absent: dt known or assume 1/sample index
%     if ~exist('time','var')
%         fs = 200; % <-- set sample rate if known
%         N = numel(noisy_vec);
%         time = (0:N-1)'/fs;
%     end
%     gx_noisy = noisy_vec; gx_GT = gt_vec;
% end

% -------------------------
% 3) Set options & call
% -------------------------
opts = struct();
opts.medwin = 11;
opts.edge_k = 8;
opts.maxlag_xcorr = 300;
opts.do_fractional = true;
opts.remove_dc = true;
opts.min_plateau_len = 20;
% other optional tuneables
opts.min_event_sep = 5;
opts.max_match_tol = 100;

[noisy_al, gt_al, time_al, info] = preprocessmat(gx_noisy, gx_GT, time, opts);

% -------------------------
% 4) Create timeseries for Simulink / workspace
% -------------------------
ts_noisy_al = timeseries(noisy_al, time_al);   % single(noisy_al) for fp32
ts_gt_al    = timeseries(gt_al,    time_al);   % default time double precision as fp64
assignin('base','ts_noisy_al', ts_noisy_al);
assignin('base','ts_gt_al',    ts_gt_al);
disp('Timeseries objects created and assigned to base workspace: ts_noisy_al, ts_gt_al');

% trivia...
% if it's fp32, then to make it fp64 --> double(var)
% if it's fp64, then to make it fp32 --> single(var)

% -------------------------
% 5) Quick diagnostics plots & numbers
% -------------------------
figure('Name','Preprocess diagnostics','NumberTitle','off','Position',[200 200 900 600]);
subplot(3,1,1);
plot(time, gx_noisy, '-','DisplayName','noisy raw'); hold on;
plot(time, gx_GT,    '-','DisplayName','gt raw'); hold off;
legend('Location','best'); title('Raw signals (pre-alignment)');

subplot(3,1,2);
plot(time_al, noisy_al,'b-','DisplayName','noisy aligned'); hold on;
plot(time_al, gt_al,'r-','DisplayName','gt aligned'); hold off;
legend('Location','best'); title(sprintf('Aligned signals (RMSE before %.4g -> after %.4g)', info.rmse_before, info.rmse_after));

subplot(3,1,3);
plot(time_al, noisy_al - gt_al); title('Residual (noisy - gt)'); xlabel('time');

% compute simple metrics
fprintf('RMSE before: %.6g\n', info.rmse_before);
fprintf('RMSE after : %.6g\n', info.rmse_after);
fprintf('Final integer lag: %d (fractional delta %.4g)\n', info.final_lag, info.fractional_delta);
fprintf('DC subtracted (noisy - gt): %.6g\n', info.dc_subtracted);

% -------------------------
% 6) (Optional) Save results
% -------------------------
save('preprocessed_signals.mat','noisy_al','gt_al','time_al','info','-v7.3');
disp('Saved preprocessed_signals.mat');

% -------------------------
% 7) If you want to split back into windows (optional)
% -------------------------
% If original windows were LxW and stacked column-wise:
% L = original_window_length;
% W = floor(numel(noisy_al)/L);
% wins_noisy_al = reshape(noisy_al(1:L*W), L, W);
% wins_gt_al    = reshape(gt_al(1:L*W),    L, W);