If Not Exist dev\Scripts\activate.bat Goto dev

If Exist dev\Scripts\activate.bat Goto end

:dev
pip3 install virtualenv
virtualenv dev
goto end

:end
git config core.hooksPath .githooks
call dev\Scripts\activate.bat
pip3 install -r requirements.txt