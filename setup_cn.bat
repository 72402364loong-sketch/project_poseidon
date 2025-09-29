@echo off
chcp 936 >nul
REM Project Poseidon v3.0 Windows 环境配置脚本

echo 🚀 Project Poseidon v3.0 环境配置
echo ================================================

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或未添加到PATH
    echo 请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo ✅ Python已安装
python --version

REM 创建虚拟环境
echo.
echo 📦 创建虚拟环境...
if exist poseidon_env (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv poseidon_env
    if errorlevel 1 (
        echo ❌ 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
)

REM 激活虚拟环境
echo.
echo 🔄 激活虚拟环境...
call poseidon_env\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败
    pause
    exit /b 1
)

REM 升级pip
echo.
echo ⬆️ 升级pip...
python -m pip install --upgrade pip

REM 安装PyTorch (CUDA版本)
echo.
echo 🔧 安装PyTorch...
echo 尝试安装CUDA 11.8版本...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ⚠️ CUDA 11.8版本安装失败，尝试CUDA 12.1版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo ⚠️ CUDA版本安装失败，尝试CPU版本...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        if errorlevel 1 (
            echo ❌ PyTorch安装失败
            pause
            exit /b 1
        )
    )
)

REM 安装项目依赖
echo.
echo 📦 安装项目依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 项目依赖安装失败
    pause
    exit /b 1
)

REM 创建目录
echo.
echo 📁 创建项目目录...
if not exist data mkdir data
if not exist data\urpc mkdir data\urpc
if not exist data\representation mkdir data\representation
if not exist data\classification mkdir data\classification
if not exist data\policy mkdir data\policy
if not exist outputs mkdir outputs
if not exist outputs\logs mkdir outputs\logs
if not exist outputs\checkpoints mkdir outputs\checkpoints
if not exist outputs\results mkdir outputs\results

REM 验证安装
echo.
echo 🧪 验证安装...
python -c "import torch; print('✅ PyTorch:', torch.__version__); print('✅ CUDA可用:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ❌ PyTorch验证失败
    pause
    exit /b 1
)

python -c "from models.representation_model import RepresentationModel; print('✅ 项目模块导入成功')"
if errorlevel 1 (
    echo ❌ 项目模块验证失败
    pause
    exit /b 1
)

echo.
echo 🎉 环境配置完成！
echo.
echo 下一步:
echo 1. 准备数据集 (参考 SETUP_GUIDE.md)
echo 2. 配置YAML文件
echo 3. 开始训练:
echo    python main_finetune_vision_on_urpc.py --config configs/stage0_vision_finetune.yaml
echo.
echo 要激活虚拟环境，请运行:
echo    poseidon_env\Scripts\activate.bat
echo.
pause
