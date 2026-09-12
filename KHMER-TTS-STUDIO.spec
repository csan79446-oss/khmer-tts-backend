# Build with: pyinstaller KHMER-TTS-STUDIO.spec
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('app')


a = Analysis(['app/main.py'], pathex=['.'], hiddenimports=hiddenimports)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, name='KHMER-TTS-STUDIO', console=False, icon='icon.ico')
