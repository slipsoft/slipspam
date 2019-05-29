If Not Exist dev\Scripts\activate.bat Goto dev

Goto end

:dev
pip3 install virtualenv
virtualenv dev
Goto end

:py3
cp dev\Scripts\python.exe dev\Scripts\python3.exe
Goto end

:end
If Not Exist dev\Scripts\python3.exe Goto py3
git config core.hooksPath .githooks
call dev\Scripts\activate.bat
pip3 install -r requirements.txt