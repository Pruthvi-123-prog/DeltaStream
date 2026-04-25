import deltastream, os
print(deltastream.__file__)
print(os.listdir(os.path.dirname(deltastream.__file__)))
# inspect AutoModel
from deltastream import AutoModel
import inspect
print(inspect.getsourcefile(AutoModel))
