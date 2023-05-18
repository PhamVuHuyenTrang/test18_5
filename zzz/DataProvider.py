from StreamGenerator import StreamGenerator


class DataProvider:

    def generate_drift_stream(self, n_chunks, chunk_size, n_classes, n_drifts,
                              random_state=1410):
        args = {
            "n_classes": n_classes,
            "n_drifts": n_drifts,
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "random_state": random_state,
            "n_informative": 2,
            "n_redundant": 0,
            "n_repeated": 0,
            "n_features": 2,
            "recurring": False,
            "n_clusters_per_class": 2,
        }
        stream = StreamGenerator(**args)
        return stream
