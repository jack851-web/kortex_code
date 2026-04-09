import collections

# Handle Python 3.10+ compatibility for collections ABCs
if not hasattr(collections, 'MutableSequence'):
    import collections.abc
    collections.MutableSequence = collections.abc.MutableSequence

# List of deprecated collections ABCs that moved to collections.abc
missing_attrs = [
    'MutableMapping', 'MutableSet', 'MutableSequence',
    'Mapping', 'Set', 'Sequence', 'ByteString'
]

# Ensure all required attributes are available in collections
for attr in missing_attrs:
    if not hasattr(collections, attr):
        import collections.abc
        setattr(collections, attr, getattr(collections.abc, attr))

from .config_gen3_lite import Gen3LiteConfig
from .gen3_lite import Gen3Lite, JOINT_LIMITS

__all__ = ["Gen3LiteConfig", "Gen3Lite", "JOINT_LIMITS"]