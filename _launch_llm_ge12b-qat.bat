rem @echo off
chcp 65001

set PORT=8434
set CUDA_DEVICE=0
set "MODEL=~/llama/gemma-4-12B-it-qat-GGUF/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
set "MMPROJ=~/llama/gemma-4-12B-it-qat-GGUF/mmproj-F32.gguf"

wsl -- bash /home/dodobuntu/llama/start_llama.sh "%MODEL%" "%MMPROJ%" %PORT% %CUDA_DEVICE%

pause