import sys
import collections

# Fix for AttributeError: module 'collections' has no attribute 'MutableSequence'
# This issue occurs with newer Python versions where MutableSequence was moved to collections.abc
if not hasattr(collections, 'MutableSequence'):
    import collections.abc
    collections.MutableSequence = collections.abc.MutableSequence

# Also fix other potential missing attributes for complete compatibility
missing_attrs = [
    'MutableMapping', 'MutableSet', 'MutableSequence',
    'Mapping', 'Set', 'Sequence', 'ByteString'
]

for attr in missing_attrs:
    if not hasattr(collections, attr):
        import collections.abc
        setattr(collections, attr, getattr(collections.abc, attr))