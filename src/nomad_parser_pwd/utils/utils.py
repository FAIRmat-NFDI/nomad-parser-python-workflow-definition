#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import functools
import inspect
from typing import TYPE_CHECKING, Any

from nomad.utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from structlog.stdlib import BoundLogger

DEFAULT_LOGGER = get_logger(__name__)


def log(
    function: 'Callable' = None,
    logger: 'BoundLogger' = DEFAULT_LOGGER,
    exc_msg: str = None,
    exc_raise: bool = False,
    default: Any = None,
):
    """
    Function decorator to log exceptions.

    Args:
        function (Callable): function to evaluate
        logger (Logger, optional): logger to attach exceptions
        exc_msg (str, optional): prefix to exception
        exc_raise (bool, optional): if True will raise error
        default (Any, optional): return value of function if error
    """

    def _log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = kwargs.get('logger', logger)
            _exc_msg = kwargs.get(
                'exc_msg', exc_msg or f'Exception raised in {func.__name__}:'
            )
            _exc_raise = kwargs.get('exc_raise', exc_raise)
            func.__annotations__['logger'] = _logger
            try:
                return func(
                    *args,
                    **{
                        key: val
                        for key, val in kwargs.items()
                        if key in inspect.signature(func).parameters
                    },
                )
            except Exception as e:
                _logger.warning(f'{_exc_msg} {e}')
                if _exc_raise:
                    raise e
                return kwargs.get('default', default)

        return wrapper

    return _log(function) if function else _log
