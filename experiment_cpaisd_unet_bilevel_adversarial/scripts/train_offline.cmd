@echo off
setlocal
set "W=C:\Users\nidal\Documents\Codex\2026-11-23\https-github-com-adityagupta-debug-dip"
set "MODE=%~1"
set "NAME="
if /i "%MODE%"=="pilot" set "NAME=pilot_bilevel_b18"
if /i "%MODE%"=="pilot" set "RUN_ARGS=--epochs 1 --max-train-files 72 --max-validation-attack-files 18 --max-eval-images-per-family 2"
if /i "%MODE%"=="full" set "NAME=bilevel_adversarial_w24_e30_b18"
if /i "%MODE%"=="full" set "RUN_ARGS=--epochs 30 --max-train-files 0 --max-eval-images-per-family 0"
if not defined NAME echo Usage: train_offline.cmd pilot or train_offline.cmd full
if not defined NAME exit /b 2
set "LOG=%W%\outputs\%NAME%.log"
set "PYTHONIOENCODING=utf-8"
echo Training is running locally. Keep this computer on. Log: %LOG%
"%W%\work\baseline-env\Scripts\python.exe" -u "%W%\work\run-wrappers\run_bilevel_cached.py" --train "C:\Users\nidal\Research Project\dataset\dataset\train" --validation "C:\Users\nidal\Research Project\dataset\dataset\val" --test "%W%\work\cpaisd\dataset\test" --baseline-checkpoint "%W%\outputs\retrained_baseline_30ep\unet_localizer.pt" --output "%W%\outputs\%NAME%" --seed 42 --batch-size 18 --base-width 24 --train-pgd-steps 7 --epsilon 0.02 --step-size 0.004 --positive-weight 3 --bilevel-update-interval 8 --evaluation-pgd-steps 10 --device cuda %RUN_ARGS% >> "%LOG%" 2>&1
set "CODE=%ERRORLEVEL%"
>> "%LOG%" echo [exit] code=%CODE%
exit /b %CODE%
