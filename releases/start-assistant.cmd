@echo off
"%~dp0.venv\Scripts\auction-vision-assistant.exe" --models "%~dp0models\latest-model-v6-v2.0.0-beta.7" --treasures "%~dp0data\v1\core\treasures.csv" %*
exit /b %errorlevel%
