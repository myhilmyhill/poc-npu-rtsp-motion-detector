pushd "$env:USERPROFILE\qairt\2.44.0.260225\bin"
.\envsetup.ps1 
& "$env:QAIRT_SDK_ROOT\bin\check-windows-dependency.ps1"
& "$env:QAIRT_SDK_ROOT\bin\envcheck.ps1" -m
popd

py -3.10 -m venv "venv"
& "venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python "$env:QAIRT_SDK_ROOT\bin\check-python-dependency"

# QNNExecutionProvider が使えないときは以下を実行
# pip uninstall onnxruntime
# pip install onnxruntime-qnn

python .\check_qnn.py
python .\simple_demo.py
