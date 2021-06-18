# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['app.py'],
             pathex=['E:\\_4THCSE\\2nd term\\img processing\\project\\Image-main'],
             binaries=[('C:\\python3\\Lib\\site-packages\\pylsd\\lib\\win32\\x86\\liblsd.dll', '.')],
             datas=[],
			 hiddenimports=['C:\\python3\\Lib\\site-packages\\pylsd\\bindings\\lsd_ctypes.py','sklearn.utils._cython_blas', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils', 'sklearn.utils._weight_vector', 'scipy.special.cython_special', 'skimage.filters.rank.core_cy_3d'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
