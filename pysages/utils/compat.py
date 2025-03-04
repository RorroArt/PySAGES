# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from importlib import import_module

from jax.scipy import linalg

from pysages._compat import _jax_version_tuple, _plum_version_tuple

# Compatibility utils


def try_import(new_name, old_name):
    try:
        return import_module(new_name)
    except ModuleNotFoundError:
        return import_module(old_name)


# Compatibility for jax >=0.4.1

# https://github.com/google/jax/releases/tag/jax-v0.4.1
if _jax_version_tuple < (0, 4, 1):

    def check_device_array(array):
        pass

else:

    def check_device_array(array):
        if not (array.is_fully_addressable and len(array.sharding.device_set) == 1):
            err = "Support for SharedDeviceArray or GlobalDeviceArray has not been implemented"
            raise ValueError(err)


# Compatibility for jax >=0.3.15

# https://github.com/google/jax/compare/jaxlib-v0.3.14...jax-v0.3.15
# https://github.com/google/jax/pull/11546
if _jax_version_tuple < (0, 3, 15):

    def solve_pos_def(a, b):
        return linalg.solve(a, b, sym_pos="sym")

else:

    def solve_pos_def(a, b):
        return linalg.solve(a, b, assume_a="pos")


# Compatibility for plum >=2

# https://github.com/beartype/plum/pull/73
if _plum_version_tuple < (2, 0, 0):

    def dispatch_table(dispatch):
        return dispatch._functions

    is_generic_subclass = issubclass

else:
    _bt = import_module("beartype.door")

    def dispatch_table(dispatch):
        return dispatch.functions

    def is_generic_subclass(A, B):
        return _bt.TypeHint(A) <= _bt.TypeHint(B)
