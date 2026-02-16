from abc import ABC, abstractmethod
from typing import Tuple, Any
from pathlib import Path
import numpy as np

from .data_source import DataRegistry
from .logger import logger


class ParameterPlugin(ABC):
    """
    Abstract base class for risk parameters

    Each parameter plugin:
    1. Validates required data sources
    2. Processes input data
    3. Returns normalized risk raster (0-1)

    Example plugins:
    - ElevationPlugin: Low elevation = High flood risk
    - SlopePlugin: Steep slope = High landslide risk
    - ProximityPlugin: Close to river = High flood risk
    """

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize parameter plugin

        Args:
            name: Unique identifier for this parameter
            weight: Relative weight in risk calculation (0-1)
        """
        self.name = name
        self.weight = weight
        self.result: np.ndarray = None
        self.metadata: dict = {}
        self.is_exclusion: bool = False

    @abstractmethod
    def validate_requirements(self, registry: DataRegistry) -> bool:
        """
        Check if required data sources are available

        Args:
            registry: Data registry to check

        Returns:
            True if all requirements met
        """
        pass

    @abstractmethod
    def process(self,
                registry: DataRegistry,
                grid_shape: Tuple[int, int],
                transform: Any,
                crs: str) -> np.ndarray:
        """
        Process data and return normalized risk raster

        Args:
            registry: Data source registry
            grid_shape: Target raster shape (height, width)
            transform: Rasterio transform object
            crs: Target coordinate reference system

        Returns:
            Normalized risk array (0-1, float32)
        """
        pass

    def get_statistics(self) -> dict:
        """
        Calculate statistics of the result

        Returns:
            Dictionary with min, max, mean, std
        """
        if self.result is None:
            return {}

        valid = np.isfinite(self.result)
        if not valid.any():
            return {'valid_pixels': 0}

        return {
            'min': float(np.nanmin(self.result)),
            'max': float(np.nanmax(self.result)),
            'mean': float(np.nanmean(self.result)),
            'std': float(np.nanstd(self.result)),
            'valid_pixels': int(valid.sum()),
            'total_pixels': self.result.size
        }

    def normalize_percentile(self,
                             data: np.ndarray,
                             lower: float = 5,
                             upper: float = 95) -> np.ndarray:
        """
        Normalize using percentile-based scaling (robust to outliers)

        Args:
            data: Input array
            lower: Lower percentile
            upper: Upper percentile

        Returns:
            Normalized array (0-1)
        """
        valid = np.isfinite(data)
        if not valid.any():
            logger.warning(f"{self.name}: No valid data for normalization")
            return np.zeros_like(data)

        d_min, d_max = np.percentile(data[valid], [lower, upper])

        if d_max == d_min:
            logger.warning(f"{self.name}: Constant values detected")
            return np.full_like(data, 0.5)

        normalized = np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return normalized.astype(np.float32)

    def normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """
        Standard min-max normalization

        Args:
            data: Input array

        Returns:
            Normalized array (0-1)
        """
        valid = np.isfinite(data)
        if not valid.any():
            return np.zeros_like(data)

        d_min = np.nanmin(data)
        d_max = np.nanmax(data)

        if d_max == d_min:
            return np.full_like(data, 0.5)

        normalized = (data - d_min) / (d_max - d_min)
        return normalized.astype(np.float32)

    def log_processing(self):
        logger.info(f"   Processing: {self.name} (weight={self.weight:.3f})")

    def log_result(self):
        stats = self.get_statistics()
        if stats:
            logger.info(f"      Range: {stats['min']:.3f} - {stats['max']:.3f}")
        else:
            logger.warning(f"      No valid data produced")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight:.3f})"

    def __str__(self) -> str:
        return f"{self.name} ({self.weight:.3f})"


class RasterParameterPlugin(ParameterPlugin):
    """
    Mainly for normalization.

    Subclasses only need to declare:
        source_keys: Registry keys to try (first found is used)
        inverse: Whether to invert the result (1 - normalized)
        normalization: "percentile", "minmax", "rank", or "zscore"

    And optionally override:
        transform_data(data, ctx): Custom transformation before normalization

    Example:
        class ElevationPlugin(RasterParameterPlugin):
            source_keys = ['dem', 'elevation']
            inverse = True
    """

    source_keys: list[str] = []
    inverse: bool = False
    normalization: str = "percentile"

    def __init__(self, name: str = "", weight: float = 1.0, **kwargs):
        if not name:
            name = self.__class__.__name__.replace("Plugin", "").lower()
        super().__init__(name, weight)
        for key, val in kwargs.items():
            setattr(self, key, val)

    def validate_requirements(self, registry: DataRegistry) -> bool:
        return any(registry.has(k) for k in self.source_keys)

    def _find_source(self, registry: DataRegistry):
        for key in self.source_keys:
            if registry.has(key):
                return registry.get(key)
        raise KeyError(
            f"{self.name}: None of {self.source_keys} found in registry. "
            f"Available: {registry.list_sources()}"
        )

    def transform_data(self, data: np.ndarray, ctx) -> np.ndarray:
        """
        Override to apply custom transformations before normalization.

        Args:
            data: Resampled raster array
            ctx: ProcessingContext instance

        Returns:
            Transformed array
        """
        return data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        from . import normalize as norm

        match self.normalization:
            case "percentile":
                return norm.percentile(data)
            case "minmax":
                return norm.minmax(data)
            case "rank":
                return norm.rank(data)
            case "zscore":
                return norm.zscore(data)
            case _:
                return norm.percentile(data)

    def process(self, registry, grid_shape, transform, crs):
        from .processing import ProcessingContext

        self.log_processing()
        ctx = ProcessingContext(grid_shape, transform, crs)

        source = self._find_source(registry)
        data = ctx.resample_raster(source)
        data = self.transform_data(data, ctx)
        normalized = self._normalize(data)

        if self.inverse:
            normalized = 1 - normalized

        self.result = normalized.astype(np.float32)
        self.log_result()
        return self.result


class PluginRegistry:
    def __init__(self):
        self.plugins: dict[str, type[ParameterPlugin]] = {}

    def register(self, plugin_class: type[ParameterPlugin]) -> None:
        name = plugin_class.__name__
        self.plugins[name] = plugin_class
        logger.debug(f"Registered plugin: {name}")

    def get(self, name: str) -> type[ParameterPlugin]:
        if name not in self.plugins:
            raise KeyError(f"Plugin '{name}' not found. "
                           f"Available: {list(self.plugins.keys())}")
        return self.plugins[name]

    def list_plugins(self) -> list[str]:
        """List all registered plugins"""
        return list(self.plugins.keys())


plugin_registry = PluginRegistry()


class PostProcessorPlugin(ABC):
    """
    For post-processing results after rasterization and risk calculation.
    For example, vectorization, adding explainability, etc.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def post_process(self,
                     registry: DataRegistry,
                     risk_raster: np.ndarray,
                     transform: Any,
                     crs: str,
                     output_dir: Any,
                     config: Any = None):
        """
        Execute post-processing logic
        
        Args:
            registry: Data registry (to access input factors)
            risk_raster: The final calculated risk raster
            transform: Rasterio transform
            crs: CRS string
            output_dir: Path to output directory
            config: EngineConfig object (optional)
        """
        pass


class CircularDependencyError(Exception):
    pass


class DataLoaderPlugin(ABC):
    """
    Data loader plugin base class.
    
    Data loaders are used for:
    1. Loading data from various sources (files, APIs, etc.)
    2. Preprocessing and validation
    3. Registering data sources to DataRegistry
    
    Dependencies are declared via:
    - provides: List of registry keys this loader registers
    - requires: List of registry keys this loader needs
    
    LoaderRegistry uses topological sort to resolve execution order
    automatically based on these declarations.
    
    Example:
        class RiversLoader(DataLoaderPlugin):
            provides = ["river"]
            requires = []
        
        class ProximityLoader(DataLoaderPlugin):
            provides = ["river_proximity"]
            requires = ["river"]
    """

    provides: list[str] = []
    requires: list[str] = []

    def __init__(self, name: str):
        self.name = name
        self.loaded_sources: dict = {}

    @classmethod
    @abstractmethod
    def can_handle(cls, config) -> bool:
        """
        Check if this loader should run based on config

        Args:
            config: EngineConfig object

        Returns:
            True if this loader should be executed
        """
        pass

    @abstractmethod
    def load(self, registry: DataRegistry) -> dict:
        pass

    def log_loading(self, message: str):
        """Log loading progress"""
        logger.info(f"[{self.name}] {message}")

    def __repr__(self) -> str:
        deps = f", requires={self.requires}" if self.requires else ""
        return f"{self.__class__.__name__}(provides={self.provides}{deps})"


class LoaderRegistry:
    """
    Registry for managing data loader plugins with auto-discovery
    
    Features:
    - Auto-discover loader plugins from directory
    - Topological sort based on provides/requires declarations
    - Circular dependency detection
    - Skip loaders based on config conditions
    """

    def __init__(self):
        self.loaders: list[type[DataLoaderPlugin]] = []
        self._discovered = False

    def auto_discover(self, plugins_dir: Path):
        """
        Discover and register loader plugins from directory
        
        Args:
            plugins_dir: Root plugins directory to search
        """
        import importlib.util
        import inspect

        loaders_dir = plugins_dir / "loaders"

        if not loaders_dir.exists():
            logger.warning(f"Loaders directory not found: {loaders_dir}")
            return

        logger.info(f"Auto-discovering data loaders in: {loaders_dir}")

        for module_file in loaders_dir.glob("*.py"):
            if module_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugins.loaders.{module_file.stem}", module_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, DataLoaderPlugin) and
                            obj is not DataLoaderPlugin and
                            obj not in self.loaders):
                        self.loaders.append(obj)
                        logger.debug(
                            f"  Discovered: {obj.__name__} "
                            f"(provides={obj.provides}, requires={obj.requires})"
                        )

            except Exception as e:
                logger.error(f"Failed to load loader module {module_file.name}: {e}")

        self._discovered = True
        logger.info(f"Discovered {len(self.loaders)} data loader(s)")

    def register(self, loader_class: type[DataLoaderPlugin]):
        if loader_class not in self.loaders:
            self.loaders.append(loader_class)

    def _resolve_order(self, active_loaders: list[type[DataLoaderPlugin]]) -> list[type[DataLoaderPlugin]]:
        """
        Resolve execution order using topological sort (Kahn's algorithm)
        
        Args:
            active_loaders: List of loader classes to sort
            
        Returns:
            Loaders in dependency-safe execution order
            
        Raises:
            CircularDependencyError: If circular dependencies are detected
        """
        from collections import defaultdict, deque

        provider_map: dict[str, type[DataLoaderPlugin]] = {}
        for loader in active_loaders:
            for key in loader.provides:
                provider_map[key] = loader

        graph: dict[type, set[type]] = defaultdict(set)
        in_degree: dict[type, int] = {loader: 0 for loader in active_loaders}

        for loader in active_loaders:
            for req in loader.requires:
                if req in provider_map:
                    dependency = provider_map[req]
                    if dependency is not loader:
                        graph[dependency].add(loader)
                        in_degree[loader] += 1
                else:
                    logger.warning(
                        f"{loader.__name__} requires '{req}' "
                        f"but no loader provides it"
                    )

        queue = deque(
            sorted(
                [l for l in active_loaders if in_degree[l] == 0],
                key=lambda x: x.__name__
            )
        )
        ordered = []

        while queue:
            loader = queue.popleft()
            ordered.append(loader)

            for dependent in sorted(graph[loader], key=lambda x: x.__name__):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(active_loaders):
            unresolved = [l.__name__ for l in active_loaders if l not in ordered]
            raise CircularDependencyError(
                f"Circular dependency detected among: {unresolved}"
            )

        return ordered

    def load_all(self, config, registry: DataRegistry):
        """
        Execute all applicable loaders in dependency-resolved order
        
        Args:
            config: EngineConfig object
            registry: DataRegistry to register sources to
        """
        if not self._discovered:
            logger.warning("Loaders not discovered yet. Call auto_discover() first.")
            return

        active = [l for l in self.loaders if l.can_handle(config)]
        skipped = [l for l in self.loaders if not l.can_handle(config)]

        for loader_class in skipped:
            logger.debug(f"Skipping {loader_class.__name__} (conditions not met)")

        ordered = self._resolve_order(active)

        logger.info("\n" + "=" * 70)
        logger.info("DATA LOADING PHASE")
        logger.info("=" * 70)
        logger.info(f"Load order: {' -> '.join(l.__name__ for l in ordered)}")

        executed = 0
        for loader_class in ordered:
            provides_str = ', '.join(loader_class.provides) or 'none'
            requires_str = ', '.join(loader_class.requires) or 'none'
            logger.info(
                f"\n{loader_class.__name__} "
                f"(provides=[{provides_str}], requires=[{requires_str}])"
            )
            try:
                loader = loader_class(config)
                loader.load(registry)
                executed += 1
            except Exception as e:
                logger.error(f"Loader {loader_class.__name__} failed: {e}")
                import traceback
                traceback.print_exc()

        logger.info("\n" + "=" * 70)
        logger.info(f"DATA LOADING COMPLETE ({executed}/{len(ordered)} loader(s) executed)")
        logger.info("=" * 70 + "\n")

    def list_loaders(self) -> list[str]:
        return [loader.__name__ for loader in self.loaders]

    def __len__(self) -> int:
        return len(self.loaders)
