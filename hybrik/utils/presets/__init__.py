from .simple_transform import SimpleTransform
from .simple_transform_3d_smpl import SimpleTransform3DSMPL
from .simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from .simple_transform_cam import SimpleTransformCam

# Optional import that requires PyTorch3D
try:
    from .simple_transform_3d_smplx import SimpleTransform3DSMPLX
    _has_smplx = True
except ImportError:
    _has_smplx = False

__all__ = [
    'SimpleTransform', 'SimpleTransform3DSMPL', 'SimpleTransform3DSMPLCam', 'SimpleTransformCam']

if _has_smplx:
    __all__.append('SimpleTransform3DSMPLX')
