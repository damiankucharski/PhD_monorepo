from dask.distributed import Client, LocalCluster


class LocalRunner:
    def __init__(self):
        self.cluster = LocalCluster()  # Launches a scheduler and workers locally
        self.client = Client(self.cluster)  # Connect to distributed cluster and override default

    def map_function(self, func, iterables):
        futures = self.client.map(func, iterables)
        return futures

    def get_dashboard_adress(self):
        return self.client.dashboard_link

    def get_failed_futures(self, futures):
        return [future for future in futures if future.status == "error"]

    def get_successful_futures(self, futures):
        return [future for future in futures if future.status == "finished"]
