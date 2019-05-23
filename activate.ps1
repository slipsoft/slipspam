if (!(Test-Path -Path dev\Scripts\activate.ps1 -PathType Leaf)) {
	pip3 install virtualenv
	virtualenv dev
}
git config core.hooksPath .githooks
dev\Scripts\activate
pip3 install -r requirements.txt