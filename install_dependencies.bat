@echo off
echo 安裝測試所需套件...

:: 確保使用虛擬環境中的 pip
.venv\Scripts\pip install pytest pytest-cov pytest-benchmark pytest-mock pandas numpy yfinance plotly streamlit

if %ERRORLEVEL% NEQ 0 (
    echo 套件安裝失敗！
    exit /b %ERRORLEVEL%
)

echo 套件安裝完成！
pause 