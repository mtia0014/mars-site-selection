@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 如需启用真实 LLM（DeepSeek），取消下面这行注释并填入你的 Key：
rem set DEEPSEEK_API_KEY=sk-xxxx

call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
python mars_agent.py
pause
