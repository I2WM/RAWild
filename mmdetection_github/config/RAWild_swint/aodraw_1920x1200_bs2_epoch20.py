import runpy as __runpy
__ns = __runpy.run_path('{{fileDirname}}/base.py')
default_scope = __ns.get('default_scope', 'mmdet')
custom_imports = __ns.get('custom_imports', None)
globals().update(__ns['build_aodraw_config'](
    (1920, 1200),
    max_epochs=20,
    train_batch_size=2,
))
