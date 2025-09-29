@echo off
chcp 65001 >nul
REM Project Poseidon v3.0 Windows Environment Setup Script

echo Project Poseidon v3.0 Environment Setup
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python is installed
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist poseidon_env (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv poseidon_env
    if errorlevel 1 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call poseidon_env\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CUDA version)
echo.
echo Installing PyTorch...
echo Trying CUDA 11.8 version...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo CUDA 11.8 version failed, trying CUDA 12.1 version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA version failed, trying CPU version...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo PyTorch installation failed
            pause
            exit /b 1
        )
    )
)

REM Install project dependencies
echo.
echo Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Project dependencies installation failed
    pause
    exit /b 1
)

REM Create directories
echo.
echo Creating project directories...
if not exist data mkdir data
if not exist data\urpc mkdir data\urpc
if not exist data\representation mkdir data\representation
if not exist data\classification mkdir data\classification
if not exist data\policy mkdir data\policy
if not exist outputs mkdir outputs
if not exist outputs\logs mkdir outputs\logs
if not exist outputs\checkpoints mkdir outputs\checkpoints
if not exist outputs\results mkdir outputs\results

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo PyTorch verification failed
    pause
    exit /b 1
)

python -c "from models.representation_model import RepresentationModel; print('Project modules imported successfully')"
if errorlevel 1 (
    echo Project modules verification failed
    pause
    exit /b 1
)

echo.
echo Environment setup completed!
echo.
echo Next steps:
echo 1. Prepare datasets (refer to SETUP_GUIDE.md)
echo 2. Configure YAML files
echo 3. Start training:
echo    python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
echo.
echo To activate virtual environment, run:
echo    poseidon_env\Scripts\activate.bat
echo.
pause