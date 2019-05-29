if (!(Test-Path -Path dev\Scripts\activate.ps1 -PathType Leaf)) {
	pip3 install virtualenv
	virtualenv dev
}
if (!(Test-Path -Path dev\Scripts\python3.exe -PathType Leaf)) {
	cp dev\Scripts\python.exe dev\Scripts\python3.exe
}
git config core.hooksPath .githooks
dev\Scripts\activate
pip3 install -r requirements.txt