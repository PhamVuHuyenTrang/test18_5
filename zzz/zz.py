import numpy as np
from DataProvider import DataProvider

dp = DataProvider()
stream = dp.generate_drift_stream(n_chunks=1000,
                                    chunk_size=100,
                                    n_classes=2,
                                    n_drifts=6,
                                    random_state=1410)

X_all = []
y_all = []

while(not stream.is_dry()):
    xy = stream.get_chunk()
    x = xy[0]
    y = xy[1]
    X_all.append(x)
    y_all.append(y)

X_all = np.array(X_all)
y_all = np.array(y_all)

np.save('X_1000_100_6.npy', X_all)
np.save('y_1000_100_6.npy', y_all)