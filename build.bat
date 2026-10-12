@echo off
set PYTHON_FOLDER=C:\Users\matef\Python\Python3114
set PYINSTALLER_PATH=%PYTHON_FOLDER%\Scripts\pyinstaller.exe

echo Cleaning previous builds...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo Running PyInstaller...
"%PYINSTALLER_PATH%" --onefile --add-data "streamlit_app.py;." --add-data "model.pth;." ^
    --hidden-import "torch" --hidden-import "torch._C" --hidden-import "torch.nn" ^
    --hidden-import "cv2" --hidden-import "albumentations" --hidden-import "PIL" --hidden-import "skimage" --hidden-import "ezdxf" --hidden-import "streamlit_drawable_canvas" --hidden-import "streamlit_image_zoom"^
    --hidden-import "streamlit.web.cli" --additional-hooks-dir=hooks run.py --clean

if errorlevel 1 (
    echo Build failed with error code %errorlevel%
) else (
    echo Build completed successfully!
    echo Copying required files...
    copy streamlit_app.py dist\
    copy model.pth dist\
    echo You can now run dist\run.exe
)

pause

@REM @echo off
@REM set PYTHON_FOLDER=C:\Users\matef\Python\Python3114
@REM set PYINSTALLER_PATH=%PYTHON_FOLDER%\Scripts\pyinstaller.exe

@REM echo Installing required packages...
@REM pip install --upgrade pip
@REM pip install Pillow==10.3.0 opencv-python-headless==4.9.0.80 albumentations==1.3.1

@REM :: Force delete existing executable
@REM taskkill /f /im run.exe 2>nul
@REM del /f /q dist\run.exe 2>nul

@REM echo Cleaning builds...
@REM rmdir /s /q build 2>nul
@REM rmdir /s /q dist 2>nul

@REM echo Building executable...
@REM "%PYINSTALLER_PATH%" --onefile ^
@REM     --add-data "streamlit_app.py;." ^
@REM     --add-data "model.pth;." ^
@REM     --hidden-import "cv2" ^
@REM     --hidden-import "PIL" ^
@REM     --hidden-import "PIL.Image" ^
@REM     --hidden-import "PIL.ImageOps" ^
@REM     --hidden-import "albumentations" ^
@REM     --hidden-import "skimage" ^
@REM     --hidden-import "ezdxf" ^
@REM     --hidden-import "tifffile" ^
@REM     --hidden-import "streamlit_drawable_canvas" ^
@REM     --hidden-import "streamlit_image_zoom" ^
@REM     --additional-hooks-dir=hooks ^
@REM     --exclude-module "torch.distributed" ^
@REM     --exclude-module "torch.utils.tensorboard" ^
@REM     --exclude-module "scipy._lib.array_api_compat.dask" ^
@REM     --exclude-module "matplotlib.tests" ^
@REM     --exclude-module "ezdxf.addons.browser" ^
@REM     run.py --clean

@REM if errorlevel 1 (
@REM     echo Build failed with error code %errorlevel%
@REM ) else (
@REM     echo Build completed successfully!
@REM     echo Copying required files...
@REM     copy streamlit_app.py dist\
@REM     copy model.pth dist\
@REM     echo Executable is ready in: %cd%\dist\run.exe
@REM )

@REM pause