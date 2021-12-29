import pandas as pd
import cupy as cp
cp.cuda.set_allocator(None)
src = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})


print(pd.DataFrame(cp.corrcoef(src.T.values), columns=["x","y"]))
import numpy as np
print(pd.DataFrame(np.corrcoef(src.T.values), columns=["x","y"],index=["x","y"]))
