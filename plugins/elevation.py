from murasa.core.base import RasterParameterPlugin


class ElevationPlugin(RasterParameterPlugin):
    """
    Elevation-based risk

    inverse=True:  Low elevation = High risk (Flood)
    inverse=False: High elevation = High risk (Landslide on steep areas)
    """
    source_keys = ['dem', 'elevation']
    inverse = True
