@echo off
setlocal enabledelayedexpansion

echo 開始執行測試流程...

:: 設置環境變量
set PYTHONPATH=%~dp0
set TEST_ENV=development

:: 確保使用虛擬環境
if not exist .venv (
    echo 創建虛擬環境...
    python -m venv .venv
)

:: 啟動虛擬環境
call .venv\Scripts\activate

:: 安裝依賴
echo 安裝/更新依賴套件...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 清理之前的測試報告
if exist reports (
    echo 清理舊的測試報告...
    rd /s /q reports
)
mkdir reports

:: 運行測試
echo 執行測試...
python scripts/run_tests.py %*

:: 檢查測試結果
if !ERRORLEVEL! NEQ 0 (
    echo 測試失敗！
    exit /b !ERRORLEVEL!
)

echo 測試完成！
echo 請查看 reports 目錄下的測試報告

:: 退出虛擬環境
deactivate
pause 