@echo off
cd /d D:\opencode\个人投资相关\hs300
if not exist logs mkdir logs
python auto_refresh.py >> logs\auto_refresh.log 2>&1
python generate_html_report.py >> logs\auto_refresh.log 2>&1
