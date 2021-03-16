""" Public facing layer for slicer.
The little slicer that could.
"""
# TODO: Move Obj and Alias class here.

from .slicer_internal import AtomicSlicer, Alias, Obj, Dim, AliasLookup, Tracked, UnifiedDataHandler
from .slicer_internal import reduced_o, resolve_dim, unify_slice, merge_shapes


class Slicer:
    """ Provides unified slicing to tensor-like objects. """

    def __init__(self, *args, **kwargs):
        """ Wraps objects in args and provides unified numpy-like slicing.

        Currently supports (with arbitrary nesting):

        - lists and tuples
        - dictionaries
        - numpy arrays
        - pandas dataframes and series
        - pytorch tensors

        Args:
            *args: Unnamed tensor-like objects.
            **kwargs: Named tensor-like objects.

        Examples:

            Basic anonymous slicing:

            >>> from slicer import Slicer as S
            >>> li = [[1, 2, 3], [4, 5, 6]]
            >>> S(li)[:, 0:2].o
            [[1, 2], [4, 5]]
            >>> di = {'x': [1, 2, 3], 'y': [4, 5, 6]}
            >>> S(di)[:, 0:2].o
            {'x': [1, 2], 'y': [4, 5]}

            Basic named slicing:

            >>> import pandas as pd
            >>> import numpy as np
            >>> df = pd.DataFrame({'A': [1, 3], 'B': [2, 4]})
            >>> ar = np.array([[5, 6], [7, 8]])
            >>> sliced = S(first=df, second=ar)[0, :]
            >>> sliced.first
            A    1
            B    2
            Name: 0, dtype: int64
            >>> sliced.second
            array([5, 6])

        """
        self.__class__._init_slicer(self, *args, **kwargs)

    @classmethod
    def from_slicer(cls, *args, **kwargs):
        """ Alternative to SUPER SLICE
        Args:
            *args:
            **kwargs:

        Returns:

        """
        slicer_instance = cls.__new__(cls)
        cls._init_slicer(slicer_instance, *args, **kwargs)
        return slicer_instance

    @classmethod
    def _init_slicer(cls, slicer_instance, *args, **kwargs):
        # NOTE: Protected attributes.
        slicer_instance._max_dim = 0
        slicer_instance._shape = tuple()

        # NOTE: Private attributes.
        slicer_instance._anon = []
        slicer_instance._objects = {}
        slicer_instance._aliases = {}
        slicer_instance._dims = {}
        slicer_instance._alias_lookup = None

        # Go through unnamed objects / aliases
        slicer_instance.__setattr__("o", args)

        # Go through named objects / aliases / dims
        for key, value in kwargs.items():
            slicer_instance.__setattr__(key, value)

        # Generate default aliases only if one object and no aliases exist
        objects_len = len(slicer_instance._objects)
        anon_len = len(slicer_instance._anon)
        aliases_len = len(slicer_instance._aliases)
        if ((objects_len == 1) ^ (anon_len == 1)) and aliases_len == 0:
            obj = None
            for _, t in slicer_instance._iter_tracked():
                obj = t

            generated_aliases = UnifiedDataHandler.default_alias(obj.o)
            for generated_alias in generated_aliases:
                slicer_instance.__setattr__(generated_alias._name, generated_alias)

    def __getitem__(self, item):
        index_tup = unify_slice(item, self._max_dim, self._alias_lookup)
        new_args = []
        new_kwargs = {}
        for name, tracked in self._iter_tracked(include_aliases=True):
            if len(tracked.dim) == 0:  # No slice on empty dim
                new_tracked = tracked
            else:
                index_slicer = AtomicSlicer(index_tup, max_dim=1)
                slicer_index = index_slicer[tracked.dim]
                sliced_o = tracked[slicer_index]
                sliced_dim = resolve_dim(index_tup, tracked.dim, sliced_o)

                new_tracked = tracked.__class__(sliced_o, sliced_dim)
                new_tracked._name = tracked._name

            if name == "o":
                new_args.append(new_tracked)
            else:
                new_kwargs[name] = new_tracked

        keep_dim_indices = [i for i, x in enumerate(index_tup) if not isinstance(x, int)]
        is_empty = len(keep_dim_indices) == 0
        for name, dim in self._dims.items():
            if is_empty:
                sliced_dim_o = []
            else:
                sliced_dim_o = AtomicSlicer(dim.o, max_dim=1)[keep_dim_indices]
            sliced_dim = Dim(sliced_dim_o)
            # NOTE: We can't have Dim in anonymous slicers, earlier checks should catch it.
            new_kwargs[name] = sliced_dim

        return self.__class__.from_slicer(*new_args, **new_kwargs)

    def __getattr__(self, item):
        """ Override default getattr to return tracked attribute.

        Args:
            item: Name of tracked attribute.
        Returns:
            Corresponding object.
        """
        if item.startswith("_"):
            return super(Slicer, self).__getattr__(item)

        if item == "shape":
            return self._shape

        if item == "o":
            return reduced_o(self._anon)
        else:
            entry = self._objects.get(item, None)
            if entry is None:
                entry = self._aliases.get(item, None)
            if entry is None:
                entry = self._dims.get(item, None)

            if entry is None:
                raise AttributeError(f"Attribute '{item}' does not exist.")

            return entry.o

    def _clear_entries(self, key):
        if key in self._objects:
            del self._objects[key]

        if key in self._aliases:
            if self._alias_lookup is not None:
                self._alias_lookup.delete(self._aliases[key])
            del self._aliases[key]

        if key in self._dims:
            del self._dims[key]

    def __setattr__(self, key, value):
        """ Override default setattr to sync tracking of slicer.

        Args:
            key: Name of tracked attribute.
            value: Either an Obj, Alias or Python Object.
        """
        if key.startswith("_"):
            return super(Slicer, self).__setattr__(key, value)

        if key == "shape":
            raise ValueError("Cannot re-assign shape. What did you think would happen?")

        # For existing attributes, honor Alias status and dimension unless specified otherwise
        if getattr(self, key, None) is not None and key != "o":
            old_obj = self._objects.get(key, None)
            old_alias = self._aliases.get(key, None)
            old_dim = self._dims.get(key, None)
            if not isinstance(value, (Tracked, Dim)):
                if old_obj:
                    value = Obj(value, dim=old_obj.dim)
                elif old_alias:
                    value = Alias(value, dim=old_alias.dim)
                elif old_dim:
                    value = Dim(value)

        if isinstance(value, Alias):
            self._clear_entries(key)
            value._name = key
            self._aliases[key] = value

            # Build lookup (for perf)
            if self._alias_lookup is None:
                self._alias_lookup = AliasLookup(self._aliases)
            else:
                self._alias_lookup.update(value)
            super(Slicer, self).__setattr__(key, value.o)
        elif isinstance(value, Dim):
            if key == "o":
                raise ValueError("Cannot have Dim within unnamed attributes.")  # pragma: no cover
            else:
                self._clear_entries(key)
                self._dims[key] = value
                super(Slicer, self).__setattr__(key, value.o)
        else:
            if key == "o":
                tracked = [Obj(x) if not isinstance(x, Obj) else x for x in value]
                self._anon = tracked
                for t in tracked:
                    self._update_stats(t)

                os = reduced_o(self._anon)
                super(Slicer, self).__setattr__(key, os)
            else:
                self._clear_entries(key)

                value = Obj(value) if not isinstance(value, Obj) else value
                value._name = key
                self._objects[key] = value
                self._update_stats(value)
                super(Slicer, self).__setattr__(key, value.o)

    def __delattr__(self, item):
        """ Override default delattr to remove tracked attribute.

        Args:
            item: Name of tracked attribute to delete.
        """
        if item.startswith("_"):
            return super(Slicer, self).__delattr__(item)

        if item == "shape":
            raise AttributeError("Cannot delete shape. What did you think would happen?")

        # Sync private attributes that help track
        self._clear_entries(item)
        if item == "o":
            self._anon.clear()

        # Recompute max_dim
        self._recompute_stats()

        # Recompute alias lookup
        # NOTE: This doesn't use diff-style deletes, but we don't care (not a perf target).
        self._alias_lookup = AliasLookup(self._aliases)

        # Update autocomplete
        super(Slicer, self).__delattr__(item)

    def __repr__(self):
        """ Override default repr for human readability.

        Returns:
            String to display.
        """
        orig = self.__dict__
        di = {}
        for key, value in orig.items():
            if not key.startswith("_"):
                di[key] = value
        return f"{self.__class__.__name__}({str(di)})"

    def _update_stats(self, tracked):
        self._max_dim = max(self._max_dim, max(tracked.dim, default=-1) + 1)
        self._shape = self._merge_shape(tracked)

    def _pad_shape(self, shape, dim):
        # Padding
        if len(dim) == 0:
            return tuple(['_' for _ in range(self._max_dim)])

        left_pad_len = min(dim)
        right_pad_len = self._max_dim - (max(dim) + 1)
        padded_shape = \
            ['_'] * left_pad_len + list(shape) + ['_'] * right_pad_len
        padded_shape = tuple(padded_shape)
        return padded_shape

    def _merge_shape(self, tracked):
        padded_current_shape = self._pad_shape(tracked.shape, tracked.dim)
        padded_parent_shape = self._pad_shape(
            self._shape,
            [x for x in range(len(self._shape))]
        )

        resolved_shape = ['_'] * self._max_dim
        for i in range(self._max_dim):
            for padded_shape in [padded_current_shape, padded_parent_shape]:
                try:
                    parent = resolved_shape[i]
                    current = padded_shape[i]

                    if current == '_' or parent == current:
                        continue
                    elif parent == '_':
                        resolved_shape[i] = current
                    elif parent != current:
                        resolved_shape[i] = None
                    else:  # pragma: no cover
                        raise ValueError("Resolve shape failure.")
                except Exception as e:
                    raise e

        return tuple(resolved_shape)

    def _recompute_stats(self):
        self._max_dim = max(
            [max(t.dim, default=-1) + 1 for _, t in self._iter_tracked()], default=0
        )
        for _, t in self._iter_tracked():
            self._shape = self._merge_shape(t)

    def _iter_tracked(self, include_aliases=False):
        for tracked in self._anon:
            yield "o", tracked
        for name, tracked in self._objects.items():
            yield name, tracked
        if include_aliases:
            for name, tracked in self._aliases.items():
                yield name, tracked
