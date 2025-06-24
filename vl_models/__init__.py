import os
import importlib
import pkgutil
import inspect

__all_classes__ = {}

package_name = __name__
package_path = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_path]):
    if module_name.startswith('_'):
        continue
    module = importlib.import_module(f"{package_name}.{module_name}")
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == f"{package_name}.{module_name}":
            __all_classes__[name] = obj

globals().update(__all_classes__)
__all__ = list(__all_classes__.keys())
